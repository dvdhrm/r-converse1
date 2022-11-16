/// Integer Handling
///
/// This module provides abstractions over integers as well as a set of utility
/// classes that implement integer operations. It is mostly a losely coupled
/// set of extensions to the standard library.

/// Internal Abstraction over Primitive Integers
///
/// An internal abstraction over all primitive integers in the Rust standard
/// library. This allows implementing other traits for all primitive integers
/// without requiring hard-coding implementations for each primitive type.
/// Instead, this trait implements all the required interfaces on all primitive
/// integers once, and then others can rely on this.
///
/// There has always been efforts to put something similar into the Rust
/// standard library, but so far these efforts have not been stabilised. Hence,
/// we carry this internal helper.
///
/// Safety
/// ------
///
/// This trait requires the implementor to guarantee that the underlying type
/// can be transmuted from a properly aligned and sized memory block of any
/// byte content. This most likely means the type must not contain padding or
/// other special bit markers.
unsafe trait PrimInt: Copy {
    fn from_be(x: Self) -> Self;
    fn from_le(x: Self) -> Self;
    fn to_be(self) -> Self;
    fn to_le(self) -> Self;
}

// Implement `PrimInt` on all primitive integers by simply mapping to the
// functions provided by the standard library.
macro_rules! impl_primint {
    ( $t:ident ) => {
        unsafe impl PrimInt for $t {
            fn from_be(x: Self) -> Self { $t::from_be(x) }
            fn from_le(x: Self) -> Self { $t::from_le(x) }
            fn to_be(self) -> Self { self.to_be() }
            fn to_le(self) -> Self { self.to_le() }
        }
    }
}

impl_primint!(u8);
impl_primint!(u16);
impl_primint!(u32);
impl_primint!(u64);
impl_primint!(u128);
impl_primint!(usize);
impl_primint!(i8);
impl_primint!(i16);
impl_primint!(i32);
impl_primint!(i64);
impl_primint!(i128);
impl_primint!(isize);

/// Types of Foreign Endianness
///
/// This trait allows converting types from foreign byte orders to the
/// machine-native byte-order and vice versa. The trait is meant for immutable
/// and copyable types (in particular integers). Bigger or more complex
/// structures are not suitable for this trait.
///
/// The idea is to define type-aliases for the most common integer types and
/// implement this trait to allow representing foreign byte orders in the type
/// system and prevent accidental misuse of non-native integers.
///
/// The trait-generic `T` defines the type of the native representation. It
/// must match the foreign representation, and the type is used to contain
/// values of both (e.g., both `to_raw()` and `to_native()` return value of
/// this type).
///
/// A common use-case is to embed such types in protocol-structures and thus
/// allow them to be mapped onto foreign-ordered memory blocks without
/// requiring conversions.
///
/// Safety
/// ------
///
/// This trait requires the implementation to guarantee its size and alignment
/// match that of `T`, and it must support transmuting from `T`. This allows
/// users to create values of this type by simply transmuting a value of type
/// `T`. Since `T` represents both foreign and native values, special care is
/// required if the memory representation of `T` contains padding or other
/// unaccounted bits!
///
/// If `T` is `Send`, implementations must guarantee that the implementing type
/// is also `Send`.
pub unsafe trait ForeignEndian<T>
    where T: Copy,
          Self: Copy,
{
    /// Create from raw value
    ///
    /// Take the raw, foreign-ordered value `raw` and create a wrapping object
    /// that protects the value from unguarded access. This must not modify the
    /// raw value in any way.
    fn from_raw(raw: T) -> Self;

    /// Return raw value
    ///
    /// Return the underlying raw, foreign-ordered value behind this wrapping
    /// object. The value must be returned without any modifications.
    fn to_raw(self) -> T;

    /// Create value from native representation
    ///
    /// Create the foreign-ordered value from a native value, converting the
    /// value before retaining it, if required.
    fn from_native(native: T) -> Self;

    /// Return native representation
    ///
    /// Return the native representation of the value behind this wrapping
    /// object. The value is to be converted to the native representation
    /// before returning it, if required.
    fn to_native(self) -> T;
}

/// Big-endian Encoded Values
///
/// Base structure that represents values encoded as big-endian. It is a
/// simple wrapping-structure with the same alignment and size requirements as
/// the type it wraps.
///
/// The base type merely provides copy and debug facilities. However, for all
/// primitive integer types this implements `ForeignEndian` and thus allows
/// conversion from and to native representation.
#[repr(transparent)]
pub struct BigEndian<T>
    where T: Copy
{
    raw: T,
}

// All `BigEndian` types are clonable.
impl<T: Copy> Clone for BigEndian<T> {
    fn clone(&self) -> BigEndian<T> {
        BigEndian { raw: self.raw }
    }
}

// All `BigEndian` types are copyable.
impl<T: Copy> Copy for BigEndian<T> {}

// For debugging simply print the raw values.
impl<T: Copy + core::fmt::Debug> core::fmt::Debug for BigEndian<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        fmt.debug_struct("BigEndian")
           .field("raw", &self.raw)
           .finish()
    }
}

unsafe impl<T> ForeignEndian<T> for BigEndian<T>
    where T: PrimInt
{
    fn from_raw(raw: T) -> Self {
        Self { raw: raw }
    }

    fn to_raw(self) -> T {
        self.raw
    }

    fn from_native(native: T) -> Self {
        Self { raw: native.to_le() }
    }

    fn to_native(self) -> T {
        T::from_le(self.raw)
    }
}

// Map the default from foreign to native.
impl<T> Default for BigEndian<T>
    where T: Copy + Default,
          Self: ForeignEndian<T>
{
    fn default() -> Self {
        Self::from_native(Default::default())
    }
}

// Convert to native for user display.
impl<T> core::fmt::Display for BigEndian<T>
    where T: Copy + core::fmt::Display,
          Self: ForeignEndian<T>
{
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        <T as core::fmt::Display>::fmt(&self.to_native(), fmt)
    }
}

// Allow import from native type.
impl<T> From<T> for BigEndian<T>
    where T: Copy,
          Self: ForeignEndian<T>,
{
    fn from(v: T) -> Self {
        Self::from_native(v)
    }
}

// Inherit partial-equality from the native type.
impl<T> PartialEq for BigEndian<T>
    where T: Copy + PartialEq,
          Self: ForeignEndian<T>,
{
    fn eq(&self, other: &Self) -> bool {
        <T as PartialEq>::eq(&self.raw, &other.raw)
    }
}

// `ForeignEndian` guarantees `Send`, so map it for the wrapper as well.
unsafe impl<T> Send for BigEndian<T>
    where T: Copy,
          Self: ForeignEndian<T>
{
}

/// Little-endian Encoded Values
///
/// Base structure that represents values encoded as little-endian. It is a
/// simple wrapping-structure with the same alignment and size requirements as
/// the type it wraps.
///
/// The base type merely provides copy and debug facilities. However, for all
/// primitive integer types this implements `ForeignEndian` and thus allows
/// conversion from and to native representation.
#[repr(transparent)]
pub struct LittleEndian<T>
    where T: Copy
{
    raw: T,
}

// All `LittleEndian` types are clonable.
impl<T: Copy> Clone for LittleEndian<T> {
    fn clone(&self) -> LittleEndian<T> {
        LittleEndian { raw: self.raw }
    }
}

// All `LittleEndian` types are copyable.
impl<T: Copy> Copy for LittleEndian<T> {}

// For debugging simply print the raw values.
impl<T: Copy + core::fmt::Debug> core::fmt::Debug for LittleEndian<T> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        fmt.debug_struct("LittleEndian")
           .field("raw", &self.raw)
           .finish()
    }
}

unsafe impl<T> ForeignEndian<T> for LittleEndian<T>
    where T: PrimInt
{
    fn from_raw(raw: T) -> Self {
        Self { raw: raw }
    }

    fn to_raw(self) -> T {
        self.raw
    }

    fn from_native(native: T) -> Self {
        Self { raw: native.to_le() }
    }

    fn to_native(self) -> T {
        T::from_le(self.raw)
    }
}

// Map the default from foreign to native.
impl<T> Default for LittleEndian<T>
    where T: Copy + Default,
          Self: ForeignEndian<T>
{
    fn default() -> Self {
        Self::from_native(Default::default())
    }
}

// Convert to native for user display.
impl<T> core::fmt::Display for LittleEndian<T>
    where T: Copy + core::fmt::Display,
          Self: ForeignEndian<T>
{
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        <T as core::fmt::Display>::fmt(&self.to_native(), fmt)
    }
}

// Allow import from native type.
impl<T> From<T> for LittleEndian<T>
    where T: Copy,
          Self: ForeignEndian<T>,
{
    fn from(v: T) -> Self {
        Self::from_native(v)
    }
}

// Inherit partial-equality from the native type.
impl<T> PartialEq for LittleEndian<T>
    where T: Copy + PartialEq,
          Self: ForeignEndian<T>,
{
    fn eq(&self, other: &Self) -> bool {
        <T as PartialEq>::eq(&self.raw, &other.raw)
    }
}

// `ForeignEndian` guarantees `Send`, so map it for the wrapper as well.
unsafe impl<T> Send for LittleEndian<T>
    where T: Copy,
          Self: ForeignEndian<T>
{
}

#[allow(non_camel_case_types)]
pub type u8be = BigEndian<u8>;
#[allow(non_camel_case_types)]
pub type u16be = BigEndian<u16>;
#[allow(non_camel_case_types)]
pub type u32be = BigEndian<u32>;
#[allow(non_camel_case_types)]
pub type u64be = BigEndian<u64>;
#[allow(non_camel_case_types)]
pub type u128be = BigEndian<u128>;
#[allow(non_camel_case_types)]
pub type i8be = BigEndian<i8>;
#[allow(non_camel_case_types)]
pub type i16be = BigEndian<i16>;
#[allow(non_camel_case_types)]
pub type i32be = BigEndian<i32>;
#[allow(non_camel_case_types)]
pub type i64be = BigEndian<i64>;
#[allow(non_camel_case_types)]
pub type i128be = BigEndian<i128>;

#[allow(non_camel_case_types)]
pub type u8le = LittleEndian<u8>;
#[allow(non_camel_case_types)]
pub type u16le = LittleEndian<u16>;
#[allow(non_camel_case_types)]
pub type u32le = LittleEndian<u32>;
#[allow(non_camel_case_types)]
pub type u64le = LittleEndian<u64>;
#[allow(non_camel_case_types)]
pub type u128le = LittleEndian<u128>;
#[allow(non_camel_case_types)]
pub type i8le = LittleEndian<i8>;
#[allow(non_camel_case_types)]
pub type i16le = LittleEndian<i16>;
#[allow(non_camel_case_types)]
pub type i32le = LittleEndian<i32>;
#[allow(non_camel_case_types)]
pub type i64le = LittleEndian<i64>;
#[allow(non_camel_case_types)]
pub type i128le = LittleEndian<i128>;
