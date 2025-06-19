> [!Tip]
> There's a great video on how precision of floating point numbers works that can be found here: [Link](https://www.youtube.com/watch?v=bbkcEiUjehk). Very much worth a watch.

## Basics of Floating Point Numbers

In Computer Science, we use floating point numbers to handle a range of values efficiently, where the name 'floating point' comes from the decimal point being able to move (or 'float'), allowing for the representation of both large and very small numbers.

### Structure of floating point numbers

![[Pasted image 20250619213153.png]]

Floating point numbers are represented using three parts:

1. **Sign Bit**: Indicates whether the number is positive or negative (0 meaning positive, 1 meaning negative).

2. **Exponent**: Tells the computer where to put the decimal point by representing how many times and which way to shift (similar to how we multiply by 10 to shift one place right in a base 10 number system).

3. **Mantissa (Significand)**: Holds the significant digits of the number, determining the precision.

### Base 10 vs Base 2

### Base 10 (Decimal)

Let's take a standard decimal number such as `42.5`.

In standard scientific notation, this can be shown as:

- `42.5 x 10^1`
  - 4.25 is the mantissa (significand)
  - 10 is the base
  - 1 is the exponent

Because we move the decimal point one to the left, the exponent is 1. If we moved it right by 1, we'd have a -1 exponent.

### Base 2 (Binary)

To represent `42.5` in base 2:

1. Convert to binary

`42.5` in binary is `101010.1`:

- 32 + 8 + 2 + 0.5 = 42.5
- (2⁵ + 2³ + 2¹ + 2⁻¹)

2. Normalise (move binary point after the first 1)

After normalisation we get `1.010101`, so we end up with `1.010101 x 2^5`.

- 1.010101 is the mantissa
- 2 is the base since we're in binary (base 2)
- 5 is the exponent (we moved the point 5 places left)

### IEEE 754 Floating Point Encoding

To actually store our normalized number (1.010101 × 2⁵) in computer memory, we use the IEEE 754 standard:

#### Exponent Bias Explained

The exponent in IEEE 754 uses a "biased" representation:

- **Where does 127 come from?** In the 32-bit (single precision) format, the exponent field is 8 bits long, which can represent unsigned values from 0 to 255.
- **Why use bias?** To handle both positive and negative exponents efficiently, IEEE 754 defines the bias as 2^(k-1) - 1, where k is the number of bits in the exponent field.

  - For 32-bit (single precision): 2^(8-1) - 1 = 2^7 - 1 = 128 - 1 = 127
  - For 64-bit (double precision): 2^(11-1) - 1 = 2^10 - 1 = 1024 - 1 = 1023

- **How it works**:
  - Stored exponent = True exponent + Bias
  - For our example: 5 + 127 = 132 (10000100 in binary)
- **Range of representable exponents**:
  - Smallest: -127 (stored as 0)
  - Largest: +128 (stored as 255, but 255 is reserved for special values)
  - Effective range: -126 to +127

This biasing mechanism allows the processor to compare floating-point numbers efficiently (larger numbers have larger exponents).

### Encoding 42.5 in IEEE 754 Single Precision (32-bit)

Let's encode our example number 42.5 in IEEE 754 single precision:

- **Sign bit**: 0 (positive number)
- **Exponent**: 5 + bias(127) = 132 = 10000100 in binary
- **Mantissa**: 010101... (we drop the leading 1, it's implied in IEEE 754)

In memory layout:

```
0 10000100 01010100000000000000000
│ │        │
│ │        └── Mantissa (23 bits)
│ └─────────── Exponent (8 bits)
└───────────── Sign bit (1 bit)
```

This is how 42.5 is actually stored in a 32-bit floating point representation.

### Common Precision Formats

- **Single Precision (FP32)**: 32 bits total

  - 1 bit for sign
  - 8 bits for exponent
  - 23 bits for mantissa

- **Double Precision (FP64)**: 64 bits total

  - 1 bit for sign
  - 11 bits for exponent
  - 52 bits for mantissa

- **Half Precision (FP16)**: 16 bits total
  - 1 bit for sign
  - 5 bits for exponent
  - 10 bits for mantissa

The choice between these formats represents a trade-off between precision and memory usage.
