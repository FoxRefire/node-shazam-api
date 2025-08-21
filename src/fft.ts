export class ComplexNumber {
    constructor(public real: number, public imag: number) {}

    add(other: ComplexNumber): ComplexNumber {
        return new ComplexNumber(this.real + other.real, this.imag + other.imag);
    }

    subtract(other: ComplexNumber): ComplexNumber {
        return new ComplexNumber(this.real - other.real, this.imag - other.imag);
    }

    multiply(other: ComplexNumber): ComplexNumber {
        return new ComplexNumber(
            this.real * other.real - this.imag * other.imag,
            this.real * other.imag + this.imag * other.real
        );
    }

    static fromPolar(magnitude: number, angle: number): ComplexNumber {
        return new ComplexNumber(magnitude * Math.cos(angle), magnitude * Math.sin(angle));
    }
}

// Pre-computed twiddle factors for common FFT sizes
const TWIDDLE_CACHE = new Map<number, ComplexNumber[]>();

function getTwiddleFactors(N: number): ComplexNumber[] {
    if (TWIDDLE_CACHE.has(N)) {
        return TWIDDLE_CACHE.get(N)!;
    }
    
    const twiddles = new Array(N / 2);
    for (let k = 0; k < N / 2; k++) {
        const angle = (-2 * Math.PI * k) / N;
        twiddles[k] = ComplexNumber.fromPolar(1, angle);
    }
    
    TWIDDLE_CACHE.set(N, twiddles);
    return twiddles;
}

// Bit reversal for in-place FFT
function bitReverse(n: number, bits: number): number {
    let result = 0;
    for (let i = 0; i < bits; i++) {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    return result;
}

// Optimized iterative in-place FFT
export function fft(input: ComplexNumber[]): ComplexNumber[] {
    const N = input.length;
    if (N <= 1) return input;
    
    // Check if N is a power of 2
    if ((N & (N - 1)) !== 0) {
        throw new Error("FFT size must be a power of 2");
    }
    
    const bits = Math.log2(N);
    const twiddles = getTwiddleFactors(N);
    
    // Create a copy to avoid modifying the input
    const result = new Array(N);
    for (let i = 0; i < N; i++) {
        result[i] = new ComplexNumber(input[i].real, input[i].imag);
    }
    
    // Bit reversal permutation
    for (let i = 0; i < N; i++) {
        const j = bitReverse(i, bits);
        if (i < j) {
            [result[i], result[j]] = [result[j], result[i]];
        }
    }
    
    // Iterative FFT
    for (let size = 2; size <= N; size *= 2) {
        const halfSize = size / 2;
        const step = N / size;
        
        for (let group = 0; group < N; group += size) {
            for (let pair = group; pair < group + halfSize; pair++) {
                const match = pair + halfSize;
                const twiddle = twiddles[(pair - group) * step];
                
                const temp = result[match].multiply(twiddle);
                result[match] = result[pair].subtract(temp);
                result[pair] = result[pair].add(temp);
            }
        }
    }
    
    return result;
}

// Ultra-optimized FFT for real input (specialized for audio processing)
export function fftRealOptimized(input: number[], output: ComplexNumber[]): void {
    const N = input.length;
    if (N <= 1) {
        for (let i = 0; i < N; i++) {
            output[i].real = input[i];
            output[i].imag = 0;
        }
        return;
    }
    
    // Convert to complex numbers in-place
    for (let i = 0; i < N; i++) {
        output[i].real = input[i];
        output[i].imag = 0;
    }
    
    // In-place FFT without object creation
    const bits = Math.log2(N);
    const twiddles = getTwiddleFactors(N);
    
    // Bit reversal permutation
    for (let i = 0; i < N; i++) {
        const j = bitReverse(i, bits);
        if (i < j) {
            const tempReal = output[i].real;
            const tempImag = output[i].imag;
            output[i].real = output[j].real;
            output[i].imag = output[j].imag;
            output[j].real = tempReal;
            output[j].imag = tempImag;
        }
    }
    
    // Iterative FFT with minimal object creation and optimized loops
    for (let size = 2; size <= N; size *= 2) {
        const halfSize = size / 2;
        const step = N / size;
        
        for (let group = 0; group < N; group += size) {
            for (let pair = group; pair < group + halfSize; pair++) {
                const match = pair + halfSize;
                const twiddle = twiddles[(pair - group) * step];
                
                // In-place complex multiplication and addition
                const tempReal = output[match].real * twiddle.real - output[match].imag * twiddle.imag;
                const tempImag = output[match].real * twiddle.imag + output[match].imag * twiddle.real;
                
                output[match].real = output[pair].real - tempReal;
                output[match].imag = output[pair].imag - tempImag;
                
                output[pair].real += tempReal;
                output[pair].imag += tempImag;
            }
        }
    }
}

// Ultra-fast FFT for real input with pre-allocated buffers (zero allocation)
export function fftRealZeroAlloc(input: Float64Array, output: ComplexNumber[]): void {
    const N = input.length;
    if (N <= 1) {
        for (let i = 0; i < N; i++) {
            output[i].real = input[i];
            output[i].imag = 0;
        }
        return;
    }
    
    // Convert to complex numbers in-place
    for (let i = 0; i < N; i++) {
        output[i].real = input[i];
        output[i].imag = 0;
    }
    
    // In-place FFT without object creation
    const bits = Math.log2(N);
    const twiddles = getTwiddleFactors(N);
    
    // Bit reversal permutation
    for (let i = 0; i < N; i++) {
        const j = bitReverse(i, bits);
        if (i < j) {
            const tempReal = output[i].real;
            const tempImag = output[i].imag;
            output[i].real = output[j].real;
            output[i].imag = output[j].imag;
            output[j].real = tempReal;
            output[j].imag = tempImag;
        }
    }
    
    // Iterative FFT with minimal object creation and optimized loops
    for (let size = 2; size <= N; size *= 2) {
        const halfSize = size / 2;
        const step = N / size;
        
        for (let group = 0; group < N; group += size) {
            for (let pair = group; pair < group + halfSize; pair++) {
                const match = pair + halfSize;
                const twiddle = twiddles[(pair - group) * step];
                
                // In-place complex multiplication and addition
                const tempReal = output[match].real * twiddle.real - output[match].imag * twiddle.imag;
                const tempImag = output[match].real * twiddle.imag + output[match].imag * twiddle.real;
                
                output[match].real = output[pair].real - tempReal;
                output[match].imag = output[pair].imag - tempImag;
                
                output[pair].real += tempReal;
                output[pair].imag += tempImag;
            }
        }
    }
}

// Specialized FFT for real input (optimized for audio processing)
export function fftReal(input: number[]): ComplexNumber[] {
    const N = input.length;
    if (N <= 1) {
        return input.map(x => new ComplexNumber(x, 0));
    }
    
    // Convert to complex numbers
    const complexInput = new Array(N);
    for (let i = 0; i < N; i++) {
        complexInput[i] = new ComplexNumber(input[i], 0);
    }
    
    return fft(complexInput);
}
