import { ComplexNumber, fft } from './fft';
import { DecodedMessage, FrequencyBand, FrequencyPeak } from "./signature-format";

// Pre-computed Hanning window to avoid runtime calculations
const HANNING_MATRIX = new Float64Array(2048);
for (let i = 0; i < 2048; i++) {
    HANNING_MATRIX[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 1)) / 2049);
}

// Optimized modulo function
const pyMod = (a: number, b: number) => ((a % b) + b) % b;

// Pre-computed constants
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
    private readonly rangeCache: Map<string, number[]>;

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
        this.fftOutputBuffer = new Array(2048);
        this.magnitudeBuffer = new Float64Array(OUTPUT_SIZE);
        this.excerptBuffer = new Float64Array(2048);
        this.rangeCache = new Map();
        
        // Pre-compute range arrays
        this.rangeCache.set('-10,-3,3', this.range(-10, -3, 3));
        this.rangeCache.set('2,9,3', this.range(2, 9, 3));
        this.rangeCache.set('165,201,7', this.range(165, 201, 7));
        this.rangeCache.set('214,250,7', this.range(214, 250, 7));
    }

    private range(a: number, b: number, c: number = 1): number[] {
        const out: number[] = [];
        for(let i = a; i < b; i += c) out.push(i);
        return out;
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

        // Pre-apply Hanning window and create complex numbers
        for (let i = 0; i < 2048; i++) {
            this.fftInputBuffer[i] = new ComplexNumber(
                this.excerptBuffer[i] * HANNING_MATRIX[i], 
                0
            );
        }

        // Perform FFT with optimized implementation
        const fftResult = fft(this.fftInputBuffer);
        
        // Calculate magnitudes directly to pre-allocated buffer
        for (let i = 0; i < OUTPUT_SIZE; i++) {
            const complex = fftResult[i];
            const magnitude = (complex.imag * complex.imag + complex.real * complex.real) / MAGNITUDE_DIVISOR;
            this.magnitudeBuffer[i] = magnitude < MIN_MAGNITUDE ? MIN_MAGNITUDE : magnitude;
        }

        this.fftOutputs.append(new Float64Array(this.magnitudeBuffer));
    }

    doPeakSpreading(){
        const originLastFFT = this.fftOutputs.list[pyMod(this.fftOutputs.position - 1, this.fftOutputs.bufferSize)]!;
        const spreadLastFFT = new Float64Array(originLastFFT);
        
        // Optimized peak spreading with single pass
        for(let position = 0; position < OUTPUT_SIZE; position++){
            if(position < 1023){
                spreadLastFFT[position] = Math.max(
                    spreadLastFFT[position], 
                    spreadLastFFT[position + 1], 
                    spreadLastFFT[position + 2]
                );
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

        // Pre-computed range arrays
        const neighborOffsets = [
            ...this.rangeCache.get('-10,-3,3')!,
            -3, 1, 
            ...this.rangeCache.get('2,9,3')!
        ];
        
        const otherOffsets = [
            -53, -45, 
            ...this.rangeCache.get('165,201,7')!,
            ...this.rangeCache.get('214,250,7')!
        ];

        for(let binPosition = 10; binPosition < 1015; binPosition++){
            // Early exit conditions
            if(fftMinus46[binPosition] < MAGNITUDE_THRESHOLD || 
               fftMinus46[binPosition] < fftMinus49[binPosition - 1]) {
                continue;
            }

            // Find maximum neighbor in FFT minus 49
            let maxNeighborInFftMinus49 = 0;
            for(const neighborOffset of neighborOffsets){
                const candidate = fftMinus49[binPosition + neighborOffset];
                if(!isNaN(candidate)) {
                    maxNeighborInFftMinus49 = Math.max(candidate, maxNeighborInFftMinus49);
                }
            }
            
            if(fftMinus46[binPosition] <= maxNeighborInFftMinus49) {
                continue;
            }

            // Find maximum neighbor in other adjacent FFTs
            let maxNeighborInOtherAdjacentFFTs = maxNeighborInFftMinus49;
            for(const otherOffset of otherOffsets){
                const candidate = this.spreadFFTsOutput.list[
                    pyMod(this.spreadFFTsOutput.position + otherOffset, this.spreadFFTsOutput.bufferSize)
                ]![binPosition - 1];
                
                if(!isNaN(candidate)) {
                    maxNeighborInOtherAdjacentFFTs = Math.max(candidate, maxNeighborInOtherAdjacentFFTs);
                }
            }

            if(fftMinus46[binPosition] > maxNeighborInOtherAdjacentFFTs){
                // This is a peak. Store the peak
                const fftNumber = this.spreadFFTsOutput.written - 46;
                const currentMagnitude = fftMinus46[binPosition];
                const prevMagnitude = fftMinus46[binPosition - 1];
                const nextMagnitude = fftMinus46[binPosition + 1];

                // Optimized magnitude calculations
                const logCurrent = Math.log(Math.max(MAGNITUDE_THRESHOLD, currentMagnitude));
                const logPrev = Math.log(Math.max(MAGNITUDE_THRESHOLD, prevMagnitude));
                const logNext = Math.log(Math.max(MAGNITUDE_THRESHOLD, nextMagnitude));
                
                const peakMagnitude = logCurrent * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
                const peakMagnitudeBefore = logPrev * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
                const peakMagnitudeAfter = logNext * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
                
                const peakVariation1 = peakMagnitude * 2 - peakMagnitudeBefore - peakMagnitudeAfter;
                
                if(peakVariation1 <= 0){
                    console.log("Assert 2 failed - " + peakVariation1);
                    continue;
                }
                
                const peakVariation2 = (peakMagnitudeAfter - peakMagnitudeBefore) * 32 / peakVariation1;
                const correctedPeakFrequencyBin = binPosition * FREQUENCY_BIN_SCALE + peakVariation2;
                
                // Optimized frequency calculation
                const frequencyHz = correctedPeakFrequencyBin * FREQUENCY_SCALE_FACTOR;
                
                // Early frequency band determination
                let band: FrequencyBand;
                if(frequencyHz < 250){
                    continue;
                } else if(frequencyHz <= 520){
                    band = FrequencyBand._250_520;
                } else if(frequencyHz <= 1450){
                    band = FrequencyBand._520_1450;
                } else if(frequencyHz <= 3500){
                    band = FrequencyBand._1450_3500;
                } else if(frequencyHz <= 5500){
                    band = FrequencyBand._3500_5500;
                } else {
                    continue;
                }

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
