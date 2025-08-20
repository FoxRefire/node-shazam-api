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
