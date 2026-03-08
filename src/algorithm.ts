import { ComplexNumber, fftRealZeroAlloc } from './fft';
import { DecodedMessage, FrequencyBand, FrequencyPeak } from "./signature-format";

// ============================================================================
// CONSTANTS - FFT and Audio Processing
// ============================================================================
const SAMPLE_RATE = 16000;
const FFT_SIZE = 1024;
const OUTPUT_SIZE = 1025;
const FREQUENCY_BIN_SCALE = 64;
const FREQUENCY_SCALE_FACTOR = SAMPLE_RATE / 2 / FFT_SIZE / FREQUENCY_BIN_SCALE;

// ============================================================================
// CONSTANTS - Magnitude Processing
// ============================================================================
const MAGNITUDE_THRESHOLD = 1 / 64;
const MAGNITUDE_SCALE = 1477.3;
const MAGNITUDE_OFFSET = 6144;
const MAGNITUDE_DIVISOR = 1 << 17;
const MIN_MAGNITUDE = 0.0000000001;
const LOG_MAGNITUDE_THRESHOLD = Math.log(MAGNITUDE_THRESHOLD);

// ============================================================================
// CONSTANTS - Peak Detection
// ============================================================================
const PEAK_CONFIDENCE_THRESHOLD = 0.1;
const TEMPORAL_CONSISTENCY_THRESHOLD = 0.1;
const MIN_PEAK_INTERVAL_FFT = 3;
const MAX_PEAKS_PER_BAND = 8;
const FREQUENCY_CONTINUITY_THRESHOLD = 0.18;
const MAGNITUDE_VARIANCE_THRESHOLD = 1.55;

// ============================================================================
// CONSTANTS - Feature Flags
// ============================================================================
const ENABLE_ADVANCED_FILTERING = true;
const ENABLE_PEAK_CONFIDENCE = true;
const ENABLE_TEMPORAL_CONSISTENCY = true;
const ENABLE_PEAK_DENSITY_LIMIT = true;
const ENABLE_PEAK_INTERVAL_LIMIT = true;
const ENABLE_DYNAMIC_THRESHOLDS = true;
const ENABLE_FREQUENCY_CONTINUITY = true;
const ENABLE_MAGNITUDE_STABILITY = true;

// ============================================================================
// PRE-COMPUTED LOOKUP TABLES
// ============================================================================
// Pre-computed Hanning window to avoid runtime calculations
const HANNING_MATRIX = new Float64Array(2048);
for (let i = 0; i < 2048; i++) {
    HANNING_MATRIX[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 1)) / 2049);
}

// Dynamic thresholds for each frequency band
const FREQUENCY_BAND_THRESHOLDS: { [key: number]: number } = {
    [FrequencyBand._250_520]: 1 / 96,
    [FrequencyBand._520_1450]: 1 / 64,
    [FrequencyBand._1450_3500]: 1 / 48,
    [FrequencyBand._3500_5500]: 1 / 32
};

const FREQUENCY_BAND_WEIGHTS: { [key: number]: number } = {
    [FrequencyBand._250_520]: 1.1,
    [FrequencyBand._520_1450]: 1.0,
    [FrequencyBand._1450_3500]: 0.95,
    [FrequencyBand._3500_5500]: 0.9
};

// Pre-computed frequency band ranges for early exit optimization
const FREQUENCY_BAND_RANGES = [
    { min: 250, max: 520, band: FrequencyBand._250_520 },
    { min: 520, max: 1450, band: FrequencyBand._520_1450 },
    { min: 1450, max: 3500, band: FrequencyBand._1450_3500 },
    { min: 3500, max: 5500, band: FrequencyBand._3500_5500 }
];

// Pre-computed neighbor offsets for peak detection
const NEIGHBOR_OFFSETS = new Int32Array([-10, -7, -4, -3, 1, 2, 5, 8]);
const OTHER_OFFSETS = new Int32Array([
    -53, -45, 165, 172, 179, 186, 193, 200, 214, 221, 228, 235, 242, 249
]);

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

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
// Optimized modulo function
const pyMod = (a: number, b: number) => ((a % b) + b) % b;

// ============================================================================
// RING BUFFER IMPLEMENTATION
// ============================================================================
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

// ============================================================================
// SIGNATURE GENERATOR
// ============================================================================
export class SignatureGenerator{
    // ========================================================================
    // PRIVATE FIELDS
    // ========================================================================
    private inputPendingProcessing: number[] = [];
    private samplesProcessed: number = 0;
    private ringBufferOfSamples!: RingBuffer<number>;
    private fftOutputs!: RingBuffer<Float64Array>;
    private spreadFFTsOutput!: RingBuffer<Float64Array>;
    private nextSignature!: DecodedMessage;
    
    // Pre-allocated buffers to avoid repeated allocations
    private fftInputBuffer!: ComplexNumber[];
    private fftOutputBuffer!: ComplexNumber[];
    private magnitudeBuffer!: Float64Array;
    private excerptBuffer!: Float64Array;
    
    // Noise filtering buffer
    private noiseProfile!: Float64Array;
    private smoothedMagnitudes!: Float64Array;
    private noiseUpdateRate: number = 0.05;

    // ========================================================================
    // CONSTRUCTOR AND INITIALIZATION
    // ========================================================================
    constructor(){
        this.initFields();
        this.initializeBuffers();
    }

    private initFields(){
        this.ringBufferOfSamples = new RingBuffer<number>(2048, 0);
        this.fftOutputs = new RingBuffer<Float64Array>(256, () => new Float64Array(OUTPUT_SIZE));
        this.spreadFFTsOutput = new RingBuffer<Float64Array>(256, () => new Float64Array(OUTPUT_SIZE));
        this.nextSignature = new DecodedMessage();
        this.nextSignature.sampleRateHz = SAMPLE_RATE;
        this.nextSignature.numberSamples = 0;
        this.nextSignature.frequencyBandToSoundPeaks = {};
    }

    private initializeBuffers(){
        // Pre-allocate FFT buffers
        this.fftInputBuffer = new Array(2048);
        this.fftOutputBuffer = new Array(2048);
        for (let i = 0; i < 2048; i++) {
            this.fftInputBuffer[i] = new ComplexNumber(0, 0);
            this.fftOutputBuffer[i] = new ComplexNumber(0, 0);
        }
        
        // Pre-allocate magnitude and excerpt buffers
        this.magnitudeBuffer = new Float64Array(OUTPUT_SIZE);
        this.excerptBuffer = new Float64Array(2048);
        
        // Initialize noise filtering buffers
        this.noiseProfile = new Float64Array(OUTPUT_SIZE);
        this.smoothedMagnitudes = new Float64Array(OUTPUT_SIZE);
    }

    // ========================================================================
    // PUBLIC INTERFACE
    // ========================================================================
    // Push in chunks to avoid "Maximum call stack size exceeded" when
    // spreading very large sample arrays (e.g. long audio) into push().
    feedInput(s16leMonoSamples: number[]){
        const CHUNK = 32768;
        for (let i = 0; i < s16leMonoSamples.length; i += CHUNK) {
            this.inputPendingProcessing.push(
                ...s16leMonoSamples.slice(i, i + CHUNK)
            );
        }
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

    // ========================================================================
    // MAIN PROCESSING PIPELINE
    // ========================================================================
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

    // ========================================================================
    // FFT PROCESSING
    // ========================================================================
    doFFT(batchOf128S16leMonoSamples: number[]){
        this.updateRingBuffer(batchOf128S16leMonoSamples);
        this.createExcerpt();
        this.applyHanningWindow();
        this.performFFT();
        this.calculateMagnitudes();
        
        const finalMagnitudes = this.applyAdvancedFiltering();
        this.fftOutputs.append(new Float64Array(finalMagnitudes));
    }

    private updateRingBuffer(batchOf128S16leMonoSamples: number[]){
        const batchLength = batchOf128S16leMonoSamples.length;
        const startPos = this.ringBufferOfSamples.position;
        
        for (let i = 0; i < batchLength; i++) {
            this.ringBufferOfSamples.list[(startPos + i) % 2048] = batchOf128S16leMonoSamples[i];
        }
        
        this.ringBufferOfSamples.position = (startPos + batchLength) % 2048;
        this.ringBufferOfSamples.written += batchLength;
    }

    private createExcerpt(){
        const startPos = this.ringBufferOfSamples.position;
        let excerptIndex = 0;
        for (let i = startPos; i < startPos + 2048; i++) {
            this.excerptBuffer[excerptIndex++] = this.ringBufferOfSamples.list[i % 2048] || 0;
        }
    }

    private applyHanningWindow(){
        for (let i = 0; i < 2048; i++) {
            this.excerptBuffer[i] = this.excerptBuffer[i] * HANNING_MATRIX[i];
        }
    }

    private performFFT(){
        fftRealZeroAlloc(this.excerptBuffer, this.fftOutputBuffer);
    }

    private calculateMagnitudes(){
        for (let i = 0; i < OUTPUT_SIZE; i++) {
            const complex = this.fftOutputBuffer[i];
            const magnitude = (complex.imag * complex.imag + complex.real * complex.real) / MAGNITUDE_DIVISOR;
            this.magnitudeBuffer[i] = magnitude < MIN_MAGNITUDE ? MIN_MAGNITUDE : magnitude;
        }
    }

    private applyAdvancedFiltering(): Float64Array {
        if (ENABLE_ADVANCED_FILTERING) {
            this.updateNoiseProfile(this.magnitudeBuffer);
            const filteredMagnitudes = this.applySpectralSubtraction(this.magnitudeBuffer);
            return this.smoothMagnitudes(filteredMagnitudes);
        }
        return this.magnitudeBuffer;
    }

    // ========================================================================
    // PEAK SPREADING
    // ========================================================================
    doPeakSpreading(){
        const originLastFFT = this.fftOutputs.list[pyMod(this.fftOutputs.position - 1, this.fftOutputs.bufferSize)]!;
        const spreadLastFFT = new Float64Array(originLastFFT);
        
        this.spreadPeaks(spreadLastFFT);
        this.spreadToAdjacentFFTs(spreadLastFFT);
        
        this.spreadFFTsOutput.append(spreadLastFFT);
    }

    private spreadPeaks(spreadLastFFT: Float64Array){
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
        }
    }

    private spreadToAdjacentFFTs(spreadLastFFT: Float64Array){
        const offsets = [-1, -3, -6];
        
        for(let position = 0; position < OUTPUT_SIZE; position++){
            const maxValue = spreadLastFFT[position];
            
            for(const formerFftNum of offsets){
                const formerFftOutput = this.spreadFFTsOutput.list[
                    pyMod(this.spreadFFTsOutput.position + formerFftNum, this.spreadFFTsOutput.bufferSize)
                ]!;
                
                if(!isNaN(formerFftOutput[position])) {
                    formerFftOutput[position] = Math.max(formerFftOutput[position], maxValue);
                }
            }
        }
    }

    // ========================================================================
    // PEAK RECOGNITION
    // ========================================================================
    doPeakRecognition(){
        const fftMinus46 = this.fftOutputs.list[pyMod(this.fftOutputs.position - 46, this.fftOutputs.bufferSize)]!;
        const fftMinus49 = this.spreadFFTsOutput.list[pyMod(this.spreadFFTsOutput.position - 49, this.spreadFFTsOutput.bufferSize)]!;

        for(let binPosition = 10; binPosition < 1015; binPosition++){
            const currentMagnitude = fftMinus46[binPosition];
            
            if (!this.isValidPeakCandidate(currentMagnitude, binPosition, fftMinus46, fftMinus49)) {
                continue;
            }

            if (this.shouldProcessPeak(currentMagnitude, binPosition, fftMinus46, fftMinus49)) {
                this.processValidPeak(binPosition, currentMagnitude, fftMinus46);
            }
        }
    }

    private isValidPeakCandidate(
        currentMagnitude: number, 
        binPosition: number, 
        fftMinus46: Float64Array, 
        fftMinus49: Float64Array
    ): boolean {
        if(currentMagnitude < MAGNITUDE_THRESHOLD || 
           currentMagnitude < fftMinus49[binPosition - 1]) {
            return false;
        }

        const maxNeighborInFftMinus49 = this.findMaxNeighborInFftMinus49(binPosition, fftMinus49);
        if(currentMagnitude <= maxNeighborInFftMinus49) {
            return false;
        }

        const maxNeighborInOtherAdjacentFFTs = this.findMaxNeighborInOtherAdjacentFFTs(binPosition, fftMinus49);
        return currentMagnitude > maxNeighborInOtherAdjacentFFTs;
    }

    private findMaxNeighborInFftMinus49(binPosition: number, fftMinus49: Float64Array): number {
        let maxNeighbor = 0;
        for(const neighborOffset of NEIGHBOR_OFFSETS){
            const candidate = fftMinus49[binPosition + neighborOffset];
            if(!isNaN(candidate)) {
                maxNeighbor = Math.max(candidate, maxNeighbor);
            }
        }
        return maxNeighbor;
    }

    private findMaxNeighborInOtherAdjacentFFTs(binPosition: number, fftMinus49: Float64Array): number {
        let maxNeighbor = this.findMaxNeighborInFftMinus49(binPosition, fftMinus49);
        
        for(const otherOffset of OTHER_OFFSETS){
            const candidate = this.spreadFFTsOutput.list[
                pyMod(this.spreadFFTsOutput.position + otherOffset, this.spreadFFTsOutput.bufferSize)
            ]![binPosition - 1];
            
            if(!isNaN(candidate)) {
                maxNeighbor = Math.max(candidate, maxNeighbor);
            }
        }
        return maxNeighbor;
    }

    private shouldProcessPeak(
        currentMagnitude: number, 
        binPosition: number, 
        fftMinus46: Float64Array, 
        fftMinus49: Float64Array
    ): boolean {
        if (ENABLE_PEAK_CONFIDENCE) {
            const maxNeighborInFftMinus49 = this.findMaxNeighborInFftMinus49(binPosition, fftMinus49);
            const maxNeighborInOtherAdjacentFFTs = this.findMaxNeighborInOtherAdjacentFFTs(binPosition, fftMinus49);
            
            const peakConfidence = this.calculatePeakConfidence(
                currentMagnitude, 
                maxNeighborInFftMinus49, 
                maxNeighborInOtherAdjacentFFTs
            );
            
            if (peakConfidence < PEAK_CONFIDENCE_THRESHOLD && currentMagnitude <= MAGNITUDE_THRESHOLD * 4) {
                return false;
            }
        }
        
        if (ENABLE_TEMPORAL_CONSISTENCY) {
            const temporalConsistency = this.checkTemporalConsistency(binPosition, fftMinus46);
            if (temporalConsistency < TEMPORAL_CONSISTENCY_THRESHOLD && currentMagnitude <= MAGNITUDE_THRESHOLD * 3) {
                return false;
            }
        }
        
        if (ENABLE_FREQUENCY_CONTINUITY) {
            const frequencyContinuity = this.checkFrequencyContinuity(binPosition, fftMinus46);
            if (frequencyContinuity < FREQUENCY_CONTINUITY_THRESHOLD && currentMagnitude <= MAGNITUDE_THRESHOLD * 5) {
                return false;
            }
        }

        if (ENABLE_MAGNITUDE_STABILITY) {
            const magnitudeStability = this.checkMagnitudeStability(binPosition, fftMinus46);
            if (magnitudeStability < 0.3 && currentMagnitude <= MAGNITUDE_THRESHOLD * 4) {
                return false;
            }
        }

        return true;
    }

    private processValidPeak(binPosition: number, currentMagnitude: number, fftMinus46: Float64Array){
        const fftNumber = this.spreadFFTsOutput.written - 46;
        const prevMagnitude = fftMinus46[binPosition - 1];
        const nextMagnitude = fftMinus46[binPosition + 1];

        const peakData = this.calculatePeakData(binPosition, currentMagnitude, prevMagnitude, nextMagnitude);
        if (!peakData) return;

        const frequencyHz = peakData.frequencyHz;
        if(frequencyHz < 250 || frequencyHz > 5500) return;

        const band = FREQUENCY_BAND_LOOKUP[Math.round(frequencyHz)];
        if (band === undefined) return;

        if (!this.passesPeakLimits(band, fftNumber, currentMagnitude)) return;

        this.addPeakToSignature(band, fftNumber, peakData);
    }

    private calculatePeakData(
        binPosition: number,
        currentMagnitude: number, 
        prevMagnitude: number, 
        nextMagnitude: number
    ): { peakMagnitude: number, correctedPeakFrequencyBin: number, frequencyHz: number } | null {
        const logCurrent = Math.max(LOG_MAGNITUDE_THRESHOLD, Math.log(currentMagnitude));
        const logPrev = Math.max(LOG_MAGNITUDE_THRESHOLD, Math.log(prevMagnitude));
        const logNext = Math.max(LOG_MAGNITUDE_THRESHOLD, Math.log(nextMagnitude));
        
        const peakMagnitude = logCurrent * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
        const peakMagnitudeBefore = logPrev * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
        const peakMagnitudeAfter = logNext * MAGNITUDE_SCALE + MAGNITUDE_OFFSET;
        
        const peakVariation1 = peakMagnitude * 2 - peakMagnitudeBefore - peakMagnitudeAfter;
        if(peakVariation1 <= 0) return null;
        
        const peakVariation2 = (peakMagnitudeAfter - peakMagnitudeBefore) * 32 / peakVariation1;
        const correctedPeakFrequencyBin = binPosition * FREQUENCY_BIN_SCALE + peakVariation2;
        const frequencyHz = correctedPeakFrequencyBin * FREQUENCY_SCALE_FACTOR;
        
        return { peakMagnitude, correctedPeakFrequencyBin, frequencyHz };
    }

    private passesPeakLimits(band: FrequencyBand, fftNumber: number, currentMagnitude: number): boolean {
        const bandKey = FrequencyBand[band];
        
        if (ENABLE_PEAK_DENSITY_LIMIT && !this.checkPeakDensityLimit(bandKey, fftNumber)) {
            return false;
        }
        
        if (ENABLE_PEAK_INTERVAL_LIMIT && !this.checkPeakIntervalLimit(bandKey, fftNumber)) {
            return false;
        }
        
        if (ENABLE_DYNAMIC_THRESHOLDS) {
            const dynamicThreshold = FREQUENCY_BAND_THRESHOLDS[band];
            if (currentMagnitude < dynamicThreshold) {
                return false;
            }
        }
        
        return true;
    }

    private addPeakToSignature(band: FrequencyBand, fftNumber: number, peakData: any){
        const bandKey = FrequencyBand[band];
        let finalPeakMagnitude = Math.round(peakData.peakMagnitude);
        
        if (ENABLE_ADVANCED_FILTERING) {
            finalPeakMagnitude = Math.round(peakData.peakMagnitude * FREQUENCY_BAND_WEIGHTS[band]);
        }
        
        if(!this.nextSignature.frequencyBandToSoundPeaks[bandKey]){
            this.nextSignature.frequencyBandToSoundPeaks[bandKey] = [];
        }
        
        this.nextSignature.frequencyBandToSoundPeaks[bandKey].push(
            new FrequencyPeak(fftNumber, finalPeakMagnitude, Math.round(peakData.correctedPeakFrequencyBin), SAMPLE_RATE)
        );
    }

    // ========================================================================
    // PEAK VALIDATION METHODS
    // ========================================================================
    private calculatePeakConfidence(
        currentMagnitude: number, 
        maxNeighborInFftMinus49: number, 
        maxNeighborInOtherAdjacentFFTs: number
    ): number {
        const magnitudeDifference = currentMagnitude - maxNeighborInOtherAdjacentFFTs;
        
        if (currentMagnitude === 0) return 0;
        
        const normalizedDifference = Math.min(magnitudeDifference / currentMagnitude, 1.0);
        
        let temporalConsistency = 0;
        if (currentMagnitude > 0) {
            temporalConsistency = Math.max(0, 1 - (maxNeighborInFftMinus49 / currentMagnitude));
        }
        
        const confidence = (normalizedDifference * 0.6 + temporalConsistency * 0.4);
        return Math.max(confidence, 0.1);
    }

    private checkTemporalConsistency(binPosition: number, currentFFT: Float64Array): number {
        const checkRange = 2;
        let consistentCount = 0;
        let totalCount = 0;
        
        for (let offset = -checkRange; offset <= checkRange; offset++) {
            const checkBin = binPosition + offset;
            if (checkBin >= 0 && checkBin < OUTPUT_SIZE) {
                const currentValue = currentFFT[checkBin];
                const centerValue = currentFFT[binPosition];
                
                if (centerValue > 0) {
                    if (Math.abs(currentValue - centerValue) / centerValue < 0.5) {
                        consistentCount++;
                    }
                }
                totalCount++;
            }
        }
        
        const consistency = totalCount > 0 ? consistentCount / totalCount : 0;
        return Math.max(consistency, 0.2);
    }

    private checkPeakDensityLimit(bandKey: string, fftNumber: number): boolean {
        const peaks = this.nextSignature.frequencyBandToSoundPeaks[bandKey];
        if (!peaks || peaks.length < MAX_PEAKS_PER_BAND) {
            return true;
        }

        const recentPeaks = peaks.filter(peak => 
            Math.abs(peak.fftPassNumber - fftNumber) <= 10
        );

        return recentPeaks.length < MAX_PEAKS_PER_BAND / 2;
    }

    private checkPeakIntervalLimit(bandKey: string, fftNumber: number): boolean {
        const peaks = this.nextSignature.frequencyBandToSoundPeaks[bandKey];
        if (!peaks || peaks.length === 0) {
            return true;
        }

        const lastPeak = peaks[peaks.length - 1];
        return Math.abs(fftNumber - lastPeak.fftPassNumber) >= MIN_PEAK_INTERVAL_FFT;
    }

    private checkFrequencyContinuity(binPosition: number, currentFFT: Float64Array): number {
        const checkRange = 1;
        let continuityScore = 0;
        let totalChecks = 0;

        for (let offset = -checkRange; offset <= checkRange; offset++) {
            if (offset === 0) continue;

            const checkBin = binPosition + offset;
            if (checkBin >= 0 && checkBin < OUTPUT_SIZE) {
                const currentValue = currentFFT[checkBin];
                const centerValue = currentFFT[binPosition];

                if (centerValue > 0) {
                    const ratio = currentValue / centerValue;
                    if (ratio > 0.1 && ratio < 10.0) {
                        continuityScore++;
                    }
                }
                totalChecks++;
            }
        }

        const continuity = totalChecks > 0 ? continuityScore / totalChecks : 0;
        return Math.max(continuity, 0.1);
    }

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
        const stability = Math.max(0, 1 - averageVariance / (MAGNITUDE_VARIANCE_THRESHOLD * 2));
        
        return Math.max(stability, 0.1);
    }

    // ========================================================================
    // NOISE FILTERING AND SMOOTHING
    // ========================================================================
    private updateNoiseProfile(magnitudes: Float64Array) {
        for (let i = 0; i < OUTPUT_SIZE; i++) {
            this.noiseProfile[i] = this.noiseProfile[i] * (1 - this.noiseUpdateRate) + 
                                   magnitudes[i] * this.noiseUpdateRate;
        }
    }

    private applySpectralSubtraction(magnitudes: Float64Array): Float64Array {
        const filtered = new Float64Array(OUTPUT_SIZE);
        const alpha = 1.2;
        
        for (let i = 0; i < OUTPUT_SIZE; i++) {
            const noiseLevel = this.noiseProfile[i];
            const signalLevel = magnitudes[i];
            
            if (signalLevel > noiseLevel * alpha) {
                filtered[i] = signalLevel - noiseLevel * (alpha - 1);
            } else {
                filtered[i] = signalLevel * 0.5;
            }
            
            filtered[i] = Math.max(filtered[i], MIN_MAGNITUDE);
        }
        
        return filtered;
    }

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
