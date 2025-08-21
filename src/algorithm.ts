import { ComplexNumber, fftRealZeroAlloc } from './fft';
import { DecodedMessage, FrequencyBand, FrequencyPeak } from "./signature-format";

// Pre-computed Hanning window to avoid runtime calculations
const HANNING_MATRIX = new Float64Array(2048);
for (let i = 0; i < 2048; i++) {
    HANNING_MATRIX[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 1)) / 2049);
}

// Pre-computed constants and lookup tables
const MAGNITUDE_THRESHOLD = 1 / 64;
const MAGNITUDE_SCALE = 1477.3;
const MAGNITUDE_OFFSET = 6144;
const FREQUENCY_BIN_SCALE = 64;
const SAMPLE_RATE = 16000;
const FFT_SIZE = 1024;
const OUTPUT_SIZE = 1025;
const FREQUENCY_SCALE_FACTOR = SAMPLE_RATE / 2 / FFT_SIZE / FREQUENCY_BIN_SCALE;
const MAGNITUDE_DIVISOR = 1 << 17;
const MIN_MAGNITUDE = 0.0000000001;

// Pre-computed frequency band ranges for early exit optimization
const FREQUENCY_BAND_RANGES = [
    { min: 250, max: 520, band: FrequencyBand._250_520 },
    { min: 520, max: 1450, band: FrequencyBand._520_1450 },
    { min: 1450, max: 3500, band: FrequencyBand._1450_3500 },
    { min: 3500, max: 5500, band: FrequencyBand._3500_5500 }
];

// Pre-computed neighbor offsets for peak detection (restored to original spec)
const NEIGHBOR_OFFSETS = new Int32Array([-10, -7, -4, -3, 1, 2, 5, 8]);
const OTHER_OFFSETS = new Int32Array([
    -53, -45, 165, 172, 179, 186, 193, 200, 214, 221, 228, 235, 242, 249
]);

// Pre-computed magnitude calculation constants
const LOG_MAGNITUDE_THRESHOLD = Math.log(MAGNITUDE_THRESHOLD);

// Pre-computed frequency band lookup table for faster access
const FREQUENCY_BAND_LOOKUP = new Uint8Array(5501);
for (let freq = 250; freq <= 5500; freq++) {
    if (freq <= 520) {
        FREQUENCY_BAND_LOOKUP[freq] = FrequencyBand._250_520;
    } else if (freq <= 1450) {
        FREQUENCY_BAND_LOOKUP[freq] = FrequencyBand._520_1450;
    } else if (freq <= 3500) {
        FREQUENCY_BAND_LOOKUP[freq] = FrequencyBand._1450_3500;
    } else if (freq <= 5500) {
        FREQUENCY_BAND_LOOKUP[freq] = FrequencyBand._3500_5500;
    }
}

// Optimized modulo function
const pyMod = (a: number, b: number) => ((a % b) + b) % b;

export class RingBuffer<T> {
    public list: (T|null)[];
    public position: number = 0;
    public written: number = 0;

    constructor(public bufferSize: number, defaultValue?: T | (() => T)){
        if(typeof defaultValue === 'function'){
            this.list = Array(bufferSize).fill(null).map(defaultValue as (() => T));
        }else{
            this.list = Array(bufferSize).fill(defaultValue ?? null);
        }
    }

    append(value: T){
        this.list[this.position] = value;
        this.position = (this.position + 1) % this.bufferSize;
        this.written++;
    }
}

export class SignatureGenerator{
    private inputPendingProcessing: number[] = [];
    private samplesProcessed: number = 0;
    private ringBufferOfSamples!: RingBuffer<number>;
    private fftOutputs!: RingBuffer<Float64Array>;
    private spreadFFTsOutput!: RingBuffer<Float64Array>;
    private nextSignature!: DecodedMessage;
    
    // Pre-allocated buffers to avoid repeated allocations
    private readonly fftInputBuffer: ComplexNumber[];
    private readonly fftOutputBuffer: ComplexNumber[];
    private readonly magnitudeBuffer: Float64Array;
    private readonly excerptBuffer: Float64Array;

    private initFields(){
        this.ringBufferOfSamples = new RingBuffer<number>(2048, 0);
        this.fftOutputs = new RingBuffer<Float64Array>(256, () => new Float64Array(OUTPUT_SIZE));
        this.spreadFFTsOutput = new RingBuffer<Float64Array>(256, () => new Float64Array(OUTPUT_SIZE));
        this.nextSignature = new DecodedMessage();
        this.nextSignature.sampleRateHz = SAMPLE_RATE;
        this.nextSignature.numberSamples = 0;
        this.nextSignature.frequencyBandToSoundPeaks = {};
    }
    
    constructor(){
        this.initFields();
        
        // Pre-allocate buffers
        this.fftInputBuffer = new Array(2048);
        for (let i = 0; i < 2048; i++) {
            this.fftInputBuffer[i] = new ComplexNumber(0, 0);
        }
        this.fftOutputBuffer = new Array(2048);
        for (let i = 0; i < 2048; i++) {
            this.fftOutputBuffer[i] = new ComplexNumber(0, 0);
        }
        this.magnitudeBuffer = new Float64Array(OUTPUT_SIZE);
        this.excerptBuffer = new Float64Array(2048);
        
        // Pre-compute frequency band cache (simplified with lookup table)
        // The lookup table is now pre-computed globally
    }

    feedInput(s16leMonoSamples: number[]){
        this.inputPendingProcessing.push(...s16leMonoSamples);
    }

    getNextSignature(): DecodedMessage | null {
        if(this.inputPendingProcessing.length - this.samplesProcessed < 128){
            return null;
        }
        this.processInput(this.inputPendingProcessing);
        this.samplesProcessed += this.inputPendingProcessing.length;
        let returnedSignature = this.nextSignature;
        this.initFields();
        
        return returnedSignature;
    }

    processInput(s16leMonoSamples: number[]){
        this.nextSignature.numberSamples += s16leMonoSamples.length;
        for(let positionOfChunk = 0; positionOfChunk < s16leMonoSamples.length; positionOfChunk += 128){
            this.doFFT(s16leMonoSamples.slice(positionOfChunk, positionOfChunk + 128));
            this.doPeakSpreading();
            if(this.spreadFFTsOutput.written >= 46) {
                this.doPeakRecognition();
            }
        }
    }

    doFFT(batchOf128S16leMonoSamples: number[]){
        // Optimized ring buffer update
        const batchLength = batchOf128S16leMonoSamples.length;
        const startPos = this.ringBufferOfSamples.position;
        
        // Direct array manipulation instead of splice
        for (let i = 0; i < batchLength; i++) {
            this.ringBufferOfSamples.list[(startPos + i) % 2048] = batchOf128S16leMonoSamples[i];
        }
        
        this.ringBufferOfSamples.position = (startPos + batchLength) % 2048;
        this.ringBufferOfSamples.written += batchLength;

        // Optimized excerpt creation using pre-allocated buffer
        let excerptIndex = 0;
        for (let i = startPos; i < startPos + 2048; i++) {
            this.excerptBuffer[excerptIndex++] = this.ringBufferOfSamples.list[i % 2048] || 0;
        }

        // Apply Hanning window in-place to the FFT input buffer
        for (let i = 0; i < 2048; i++) {
            this.excerptBuffer[i] = this.excerptBuffer[i] * HANNING_MATRIX[i];
        }

        // Perform optimized FFT
        fftRealZeroAlloc(this.excerptBuffer, this.fftOutputBuffer);
        
        // Calculate magnitudes directly to pre-allocated buffer
        for (let i = 0; i < OUTPUT_SIZE; i++) {
            const complex = this.fftOutputBuffer[i];
            const magnitude = (complex.imag * complex.imag + complex.real * complex.real) / MAGNITUDE_DIVISOR;
            this.magnitudeBuffer[i] = magnitude < MIN_MAGNITUDE ? MIN_MAGNITUDE : magnitude;
        }

        this.fftOutputs.append(new Float64Array(this.magnitudeBuffer));
    }

    doPeakSpreading(){
        const originLastFFT = this.fftOutputs.list[pyMod(this.fftOutputs.position - 1, this.fftOutputs.bufferSize)]!;
        const spreadLastFFT = new Float64Array(originLastFFT);
        
        // Optimized peak spreading with single pass and early exit
        for(let position = 0; position < OUTPUT_SIZE; position++){
            if(position < 1023){
                const maxNeighbor = Math.max(
                    spreadLastFFT[position + 1], 
                    spreadLastFFT[position + 2]
                );
                if (spreadLastFFT[position] < maxNeighbor) {
                    spreadLastFFT[position] = maxNeighbor;
                }
            }

            const maxValue = spreadLastFFT[position];
            const offsets = [-1, -3, -6];
            
            for(const formerFftNum of offsets){
                const formerFftOutput = this.spreadFFTsOutput.list[
                    pyMod(this.spreadFFTsOutput.position + formerFftNum, this.spreadFFTsOutput.bufferSize)
                ]!;
                
                if(!isNaN(formerFftOutput[position])) {
                    formerFftOutput[position] = Math.max(formerFftOutput[position], maxValue);
                }
            }
        }
        this.spreadFFTsOutput.append(spreadLastFFT);
    }

    doPeakRecognition(){
        const fftMinus46 = this.fftOutputs.list[pyMod(this.fftOutputs.position - 46, this.fftOutputs.bufferSize)]!;
        const fftMinus49 = this.spreadFFTsOutput.list[pyMod(this.spreadFFTsOutput.position - 49, this.spreadFFTsOutput.bufferSize)]!;

        for(let binPosition = 10; binPosition < 1015; binPosition++){
            const currentMagnitude = fftMinus46[binPosition];
            
            // Early exit conditions with optimized checks
            if(currentMagnitude < MAGNITUDE_THRESHOLD || 
               currentMagnitude < fftMinus49[binPosition - 1]) {
                continue;
            }

            // Find maximum neighbor in FFT minus 49 using pre-computed offsets
            let maxNeighborInFftMinus49 = 0;
            for(const neighborOffset of NEIGHBOR_OFFSETS){
                const candidate = fftMinus49[binPosition + neighborOffset];
                if(!isNaN(candidate)) {
                    maxNeighborInFftMinus49 = Math.max(candidate, maxNeighborInFftMinus49);
                }
            }
            
            if(currentMagnitude <= maxNeighborInFftMinus49) {
                continue;
            }

            // Find maximum neighbor in other adjacent FFTs
            let maxNeighborInOtherAdjacentFFTs = maxNeighborInFftMinus49;
            for(const otherOffset of OTHER_OFFSETS){
                const candidate = this.spreadFFTsOutput.list[
                    pyMod(this.spreadFFTsOutput.position + otherOffset, this.spreadFFTsOutput.bufferSize)
                ]![binPosition - 1];
                
                if(!isNaN(candidate)) {
                    maxNeighborInOtherAdjacentFFTs = Math.max(candidate, maxNeighborInOtherAdjacentFFTs);
                }
            }

            if(currentMagnitude > maxNeighborInOtherAdjacentFFTs){
                // This is a peak. Store the peak
                const fftNumber = this.spreadFFTsOutput.written - 46;
                const prevMagnitude = fftMinus46[binPosition - 1];
                const nextMagnitude = fftMinus46[binPosition + 1];

                // Optimized magnitude calculations with pre-computed constants
                const logCurrent = Math.max(LOG_MAGNITUDE_THRESHOLD, Math.log(currentMagnitude));
                const logPrev = Math.max(LOG_MAGNITUDE_THRESHOLD, Math.log(prevMagnitude));
                const logNext = Math.max(LOG_MAGNITUDE_THRESHOLD, Math.log(nextMagnitude));
                
                const peakMagnitude = logCurrent * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
                const peakMagnitudeBefore = logPrev * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
                const peakMagnitudeAfter = logNext * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
                
                const peakVariation1 = peakMagnitude * 2 - peakMagnitudeBefore - peakMagnitudeAfter;
                
                if(peakVariation1 <= 0){
                    continue;
                }
                
                const peakVariation2 = (peakMagnitudeAfter - peakMagnitudeBefore) * 32 / peakVariation1;
                const correctedPeakFrequencyBin = binPosition * FREQUENCY_BIN_SCALE + peakVariation2;
                
                // Optimized frequency calculation
                const frequencyHz = correctedPeakFrequencyBin * FREQUENCY_SCALE_FACTOR;
                
                // Early frequency band determination using lookup table
                if(frequencyHz < 250 || frequencyHz > 5500){
                    continue;
                }
                
                // Use pre-computed lookup table for fastest access
                const band = FREQUENCY_BAND_LOOKUP[Math.round(frequencyHz)];
                if (band === undefined) continue;

                const bandKey = FrequencyBand[band];
                if(!this.nextSignature.frequencyBandToSoundPeaks[bandKey]){
                    this.nextSignature.frequencyBandToSoundPeaks[bandKey] = [];
                }
                
                this.nextSignature.frequencyBandToSoundPeaks[bandKey].push(
                    new FrequencyPeak(fftNumber, Math.round(peakMagnitude), Math.round(correctedPeakFrequencyBin), SAMPLE_RATE)
                );
            }
        }
    }
}
