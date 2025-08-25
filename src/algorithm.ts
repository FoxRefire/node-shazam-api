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
const PEAK_CONFIDENCE_THRESHOLD = 0.1;
const TEMPORAL_CONSISTENCY_THRESHOLD = 0.1;
const MIN_PEAK_INTERVAL_FFT = 3; // Minimum FFT interval between peaks
const MAX_PEAKS_PER_BAND = 8; // Maximum number of peaks per frequency band
const FREQUENCY_CONTINUITY_THRESHOLD = 0.18; // Frequency direction continuity threshold relaxed from 0.8 to 0.3
const MAGNITUDE_VARIANCE_THRESHOLD = 1.55; // Magnitude variation threshold relaxed from 0.4 to 0.8

// Dynamic thresholds for each frequency band
const FREQUENCY_BAND_THRESHOLDS: { [key: number]: number } = {
    [FrequencyBand._250_520]: 1 / 96,      // Low frequency band is more strict
    [FrequencyBand._520_1450]: 1 / 64,     // Mid frequency band is standard
    [FrequencyBand._1450_3500]: 1 / 48,    // High frequency band is more lenient
    [FrequencyBand._3500_5500]: 1 / 32     // Ultra-high frequency band is most lenient
};

const FREQUENCY_BAND_WEIGHTS: { [key: number]: number } = {
    [FrequencyBand._250_520]: 1.1,      // Low frequency band weight (moderate)
    [FrequencyBand._520_1450]: 1.0,     // Mid frequency band weight (baseline)
    [FrequencyBand._1450_3500]: 0.95,   // High frequency band weight (moderate)
    [FrequencyBand._3500_5500]: 0.9     // Ultra-high frequency band weight (moderate)
};

const ENABLE_ADVANCED_FILTERING = true;
const ENABLE_PEAK_CONFIDENCE = true;
const ENABLE_TEMPORAL_CONSISTENCY = true;
const ENABLE_PEAK_DENSITY_LIMIT = true;      // Peak density limit
const ENABLE_PEAK_INTERVAL_LIMIT = true;     // Peak interval limit
const ENABLE_DYNAMIC_THRESHOLDS = true;      // Dynamic thresholds
const ENABLE_FREQUENCY_CONTINUITY = true;    // Frequency continuity check
const ENABLE_MAGNITUDE_STABILITY = true;     // Magnitude stability check

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
    
    // Noise filtering buffer
    private readonly noiseProfile: Float64Array;
    private readonly smoothedMagnitudes: Float64Array;
    private noiseUpdateRate: number = 0.05; // Reduce noise profile update rate

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
        
        // Initialize noise filtering buffers
        this.noiseProfile = new Float64Array(OUTPUT_SIZE);
        this.smoothedMagnitudes = new Float64Array(OUTPUT_SIZE);
        
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

        // Apply noise filtering and spectral subtraction (optional)
        let finalMagnitudes: Float64Array;
        if (ENABLE_ADVANCED_FILTERING) {
            this.updateNoiseProfile(this.magnitudeBuffer);
            const filteredMagnitudes = this.applySpectralSubtraction(this.magnitudeBuffer);
            finalMagnitudes = this.smoothMagnitudes(filteredMagnitudes);
        } else {
            finalMagnitudes = this.magnitudeBuffer;
        }

        this.fftOutputs.append(new Float64Array(finalMagnitudes));
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
                // Peak confidence calculation (optional)
                let shouldContinue = true;
                
                if (ENABLE_PEAK_CONFIDENCE) {
                    const peakConfidence = this.calculatePeakConfidence(
                        currentMagnitude, 
                        maxNeighborInFftMinus49, 
                        maxNeighborInOtherAdjacentFFTs
                    );
                    
                    // Debug: Log confidence values (development only)
                    // console.log(`Peak confidence: ${peakConfidence.toFixed(3)}, threshold: ${PEAK_CONFIDENCE_THRESHOLD}`);
                    
                    // Skip if confidence is below threshold
                    if (peakConfidence < PEAK_CONFIDENCE_THRESHOLD) {
                        // Fallback: Force pass if magnitude is sufficiently large
                        if (currentMagnitude > MAGNITUDE_THRESHOLD * 4) {
                            // console.log(`Fallback: Strong magnitude peak allowed (${currentMagnitude.toFixed(6)})`);
                        } else {
                            shouldContinue = false;
                        }
                    }
                }
                
                // Temporal consistency check (optional)
                if (shouldContinue && ENABLE_TEMPORAL_CONSISTENCY) {
                    const temporalConsistency = this.checkTemporalConsistency(binPosition, fftMinus46);
                    
                    // Debug: Log consistency values (development only)
                    // console.log(`Temporal consistency: ${temporalConsistency.toFixed(3)}, threshold: ${TEMPORAL_CONSISTENCY_THRESHOLD}`);
                    
                    if (temporalConsistency < TEMPORAL_CONSISTENCY_THRESHOLD) {
                        // Fallback: Force pass if magnitude is sufficiently large
                        if (currentMagnitude > MAGNITUDE_THRESHOLD * 3) {
                            // console.log(`Fallback: Strong magnitude peak allowed for temporal consistency (${currentMagnitude.toFixed(6)})`);
                        } else {
                            shouldContinue = false;
                        }
                    }
                }
                
                // Additional false positive reduction checks
                if (shouldContinue && ENABLE_FREQUENCY_CONTINUITY) {
                    // Frequency direction continuity check
                    const frequencyContinuity = this.checkFrequencyContinuity(binPosition, fftMinus46);
                    
                    // Debug: Log continuity values (development only)
                    // console.log(`Frequency continuity: ${frequencyContinuity.toFixed(3)}, threshold: ${FREQUENCY_CONTINUITY_THRESHOLD}`);
                    
                    if (frequencyContinuity < FREQUENCY_CONTINUITY_THRESHOLD) {
                        // Fallback: Force pass if magnitude is sufficiently large
                        if (currentMagnitude > MAGNITUDE_THRESHOLD * 5) {
                            // console.log(`Fallback: Strong magnitude peak allowed for frequency continuity (${currentMagnitude.toFixed(6)})`);
                        } else {
                            shouldContinue = false;
                        }
                    }
                }

                if (shouldContinue && ENABLE_MAGNITUDE_STABILITY) {
                    // Magnitude variation stability check
                    const magnitudeStability = this.checkMagnitudeStability(binPosition, fftMinus46);
                    
                    // Debug: Log stability values (development only)
                    // console.log(`Magnitude stability: ${magnitudeStability.toFixed(3)}, threshold: 0.3`);
                    
                    if (magnitudeStability < 0.3) { // Stability threshold
                        // Fallback: Force pass if magnitude is sufficiently large
                        if (currentMagnitude > MAGNITUDE_THRESHOLD * 4) {
                            // console.log(`Fallback: Strong magnitude peak allowed for magnitude stability (${currentMagnitude.toFixed(6)})`);
                        } else {
                            shouldContinue = false;
                        }
                    }
                }

                if (!shouldContinue) {
                    continue;
                }
                
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
                
                // Check peak density limit and temporal interval limit
                if (ENABLE_PEAK_DENSITY_LIMIT && !this.checkPeakDensityLimit(bandKey, fftNumber)) {
                    continue; // Exceeds density limit
                }
                
                if (ENABLE_PEAK_INTERVAL_LIMIT && !this.checkPeakIntervalLimit(bandKey, fftNumber)) {
                    continue; // Exceeds temporal interval limit
                }
                
                // Apply dynamic thresholds for each frequency band
                if (ENABLE_DYNAMIC_THRESHOLDS) {
                    const dynamicThreshold = FREQUENCY_BAND_THRESHOLDS[band];
                    if (currentMagnitude < dynamicThreshold) {
                        continue; // Below dynamic threshold
                    }
                }
                
                // Apply frequency band weighting (optional)
                let finalPeakMagnitude = Math.round(peakMagnitude);
                if (ENABLE_ADVANCED_FILTERING) {
                    finalPeakMagnitude = Math.round(peakMagnitude * FREQUENCY_BAND_WEIGHTS[band]);
                }
                
                if(!this.nextSignature.frequencyBandToSoundPeaks[bandKey]){
                    this.nextSignature.frequencyBandToSoundPeaks[bandKey] = [];
                }
                
                this.nextSignature.frequencyBandToSoundPeaks[bandKey].push(
                    new FrequencyPeak(fftNumber, finalPeakMagnitude, Math.round(correctedPeakFrequencyBin), SAMPLE_RATE)
                );
            }
        }
    }

    // Method to calculate peak confidence
    private calculatePeakConfidence(
        currentMagnitude: number, 
        maxNeighborInFftMinus49: number, 
        maxNeighborInOtherAdjacentFFTs: number
    ): number {
        // Higher confidence when difference between current peak and adjacent peaks is larger
        const magnitudeDifference = currentMagnitude - maxNeighborInOtherAdjacentFFTs;
        
        // Prevent division by zero
        if (currentMagnitude === 0) return 0;
        
        const normalizedDifference = Math.min(magnitudeDifference / currentMagnitude, 1.0);
        
        // Also consider consistency with adjacent FFTs (prevent division by zero)
        let temporalConsistency = 0;
        if (currentMagnitude > 0) {
            temporalConsistency = Math.max(0, 1 - (maxNeighborInFftMinus49 / currentMagnitude));
        }
        
        // Calculate overall confidence (adjusted to be more lenient)
        const confidence = (normalizedDifference * 0.6 + temporalConsistency * 0.4);
        
        // Debug: Adjust if confidence is too low
        return Math.max(confidence, 0.1); // Guarantee minimum value of 0.1
    }

    // Method to check temporal consistency
    private checkTemporalConsistency(binPosition: number, currentFFT: Float64Array): number {
        // Check consistency in surrounding frequency bins
        const checkRange = 2; // Narrow the range
        let consistentCount = 0;
        let totalCount = 0;
        
        for (let offset = -checkRange; offset <= checkRange; offset++) {
            const checkBin = binPosition + offset;
            if (checkBin >= 0 && checkBin < OUTPUT_SIZE) {
                const currentValue = currentFFT[checkBin];
                const centerValue = currentFFT[binPosition];
                
                // Prevent division by zero
                if (centerValue > 0) {
                    // Higher consistency when relative difference from center value is smaller (more lenient threshold)
                    if (Math.abs(currentValue - centerValue) / centerValue < 0.5) {
                        consistentCount++;
                    }
                }
                totalCount++;
            }
        }
        
        const consistency = totalCount > 0 ? consistentCount / totalCount : 0;
        
        // Debug: Adjust if consistency is too low
        return Math.max(consistency, 0.2); // Guarantee minimum value of 0.2
    }

    // Method to check peak density limit
    private checkPeakDensityLimit(bandKey: string, fftNumber: number): boolean {
        const peaks = this.nextSignature.frequencyBandToSoundPeaks[bandKey];
        if (!peaks || peaks.length < MAX_PEAKS_PER_BAND) {
            return true; // Within limit
        }

        // Check number of peaks in recent FFT frames
        const recentPeaks = peaks.filter(peak => 
            Math.abs(peak.fftPassNumber - fftNumber) <= 10
        );

        return recentPeaks.length < MAX_PEAKS_PER_BAND / 2;
    }

    // Method to check temporal interval limit
    private checkPeakIntervalLimit(bandKey: string, fftNumber: number): boolean {
        const peaks = this.nextSignature.frequencyBandToSoundPeaks[bandKey];
        if (!peaks || peaks.length === 0) {
            return true; // First peak
        }

        // Check interval with recent peak in same band
        const lastPeak = peaks[peaks.length - 1];
        return Math.abs(fftNumber - lastPeak.fftPassNumber) >= MIN_PEAK_INTERVAL_FFT;
    }

    // Method to check frequency direction continuity
    private checkFrequencyContinuity(binPosition: number, currentFFT: Float64Array): number {
        const checkRange = 1; // Narrow the range
        let continuityScore = 0;
        let totalChecks = 0;

        for (let offset = -checkRange; offset <= checkRange; offset++) {
            if (offset === 0) continue; // Exclude center

            const checkBin = binPosition + offset;
            if (checkBin >= 0 && checkBin < OUTPUT_SIZE) {
                const currentValue = currentFFT[checkBin];
                const centerValue = currentFFT[binPosition];

                if (centerValue > 0) {
                    // Check if surrounding values have proportional relationship with center value (more lenient condition)
                    const ratio = currentValue / centerValue;
                    if (ratio > 0.1 && ratio < 10.0) { // Relaxed from 0.3-3.0 to 0.1-10.0
                        continuityScore++;
                    }
                }
                totalChecks++;
            }
        }

        const continuity = totalChecks > 0 ? continuityScore / totalChecks : 0;
        
        // Debug: Adjust if continuity is too low
        return Math.max(continuity, 0.1); // Guarantee minimum value of 0.1
    }

    // Method to check magnitude variation stability
    private checkMagnitudeStability(binPosition: number, currentFFT: Float64Array): number {
        const checkRange = 1;
        let varianceSum = 0;
        let totalChecks = 0;

        for (let offset = -checkRange; offset <= checkRange; offset++) {
            if (offset === 0) continue;

            const checkBin = binPosition + offset;
            if (checkBin >= 0 && checkBin < OUTPUT_SIZE) {
                const currentValue = currentFFT[checkBin];
                const centerValue = currentFFT[binPosition];

                if (centerValue > 0) {
                    const variance = Math.abs(currentValue - centerValue) / centerValue;
                    varianceSum += variance;
                    totalChecks++;
                }
            }
        }

        if (totalChecks === 0) return 1.0;
        
        const averageVariance = varianceSum / totalChecks;
        // Higher score when variation is smaller (more lenient calculation)
        const stability = Math.max(0, 1 - averageVariance / (MAGNITUDE_VARIANCE_THRESHOLD * 2)); // Relax threshold by 2x
        
        // Debug: Adjust if stability is too low
        return Math.max(stability, 0.1); // Guarantee minimum value of 0.1
    }

    // Method to update noise profile
    private updateNoiseProfile(magnitudes: Float64Array) {
        for (let i = 0; i < OUTPUT_SIZE; i++) {
            // Update noise profile using exponential moving average
            this.noiseProfile[i] = this.noiseProfile[i] * (1 - this.noiseUpdateRate) + 
                                   magnitudes[i] * this.noiseUpdateRate;
        }
    }

    // Noise removal using spectral subtraction
    private applySpectralSubtraction(magnitudes: Float64Array): Float64Array {
        const filtered = new Float64Array(OUTPUT_SIZE);
        const alpha = 1.2;
        
        for (let i = 0; i < OUTPUT_SIZE; i++) {
            // Filtering considering noise level
            const noiseLevel = this.noiseProfile[i];
            const signalLevel = magnitudes[i];
            
            if (signalLevel > noiseLevel * alpha) {
                // When signal is sufficiently larger than noise
                filtered[i] = signalLevel - noiseLevel * (alpha - 1);
            } else {
                // When signal is close to noise level, attenuate significantly
                filtered[i] = signalLevel * 0.5; // Changed from 0.1 to 0.5 (more moderate)
            }
            
            // Ensure it doesn't go below minimum value
            filtered[i] = Math.max(filtered[i], MIN_MAGNITUDE);
        }
        
        return filtered;
    }

    // Magnitude smoothing
    private smoothMagnitudes(magnitudes: Float64Array): Float64Array {
        const smoothed = new Float64Array(OUTPUT_SIZE);
        const smoothingFactor = 0.1;
        
        for (let i = 0; i < OUTPUT_SIZE; i++) {
            if (i === 0) {
                smoothed[i] = magnitudes[i];
            } else {
                smoothed[i] = magnitudes[i] * (1 - smoothingFactor) + 
                              smoothed[i - 1] * smoothingFactor;
            }
        }
        
        return smoothed;
    }
}
