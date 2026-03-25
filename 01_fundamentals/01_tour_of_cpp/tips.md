# C++ Tips & Best Practices

## 1. The Basics

### General Philosophy
- **Don't panic!** All will become clear in time.
- **Don't use built-in features exclusively.** Many fundamental features are best used indirectly through libraries, such as the ISO C++ standard library.
- **Focus on programming techniques**, not just on language features.
- **You don't have to know every detail** of C++ to write good programs.
- **The ISO C++ standard** is the final word on language definition issues.
- **Understand how language primitives map to hardware.**

### Functions
- **Package meaningful operations** as carefully named functions.
- **Single responsibility:** A function should perform a single logical operation.
- **Keep functions short.**
- **Use overloading** when functions perform conceptually the same task on different types.
- **Compile-time evaluation:**
    - If a function *may* have to be evaluated at compile time, declare it `constexpr`.
    - If a function *must* be evaluated at compile time, declare it `consteval`.
    - If a function may not have side effects, declare it `constexpr` or `consteval`.

### Variables & Types
- **Minimize scope:** Minimize the scope of a variable. Keep scopes small.
- **Avoid "magic constants";** use symbolic constants.
- **Prefer immutable data.**
- **One name per declaration:** Declare one name (only) per declaration.
- **Initialization:**
    - Prefer the `{}` initializer syntax for declarations with a named type.
    - Use `auto` to avoid repeating type names.
    - **Avoid uninitialized variables.** Don't declare a variable until you have a value to initialize it with.
- **Narrowing conversions:** Avoid narrowing conversions.
- **Digit separators:** Use digit separators to make large literals readable.
- **Unsigned types:** Use `unsigned` for bit manipulation only.
- **Pointers:**
    - Keep use of pointers simple and straightforward.
    - Use `nullptr` rather than `0` or `NULL`.

### Code Structure & Control Flow
- **Libraries:** `#include` or (preferably) `import` the libraries needed to simplify programming.
- **Complicated expressions:** Avoid complicated expressions.
- **Conditionals:** When declaring a variable in the condition of an `if`-statement, prefer the version with the implicit test against `0` or `nullptr`.
- **Loops:** Prefer range-for loops over for-loops with an explicit loop variable.

### Naming & Style
- **Name length:** Keep common and local names short; keep uncommon and nonlocal names longer.
- **Distinct names:** Avoid similar looking names.
- **Capitalization:** Avoid `ALL_CAPS` names.
- **Comments:**
    - Don't say in comments what can be clearly stated in code.
    - State intent in comments.
- **Indentation:** Maintain a consistent indentation style.

## 2. User Defined Types

### General Philosophy
- **Prefer well-defined user-defined types** over built-in types when the built-in types are too low-level.
- **Organize related data** into structures (structs or classes).

### Classes & Structs
- **Interface vs. Implementation:** Represent the distinction between an interface and an implementation using a class.
- **Structs:** A struct is simply a class with its members public by default.
- **Constructors:** Define constructors to guarantee and simplify initialization of classes.

### Enumerations
- **Use enumerations** to represent sets of named constants.
- **Prefer class enums** over "plain" enums to minimize surprises.
- **Define operations** on enumerations for safe and simple use.

### Unions & Variants
- **Avoid "naked" unions;** wrap them in a class together with a type field.
- **Prefer `std::variant`** to "naked" unions.

## 3. Modularity

### Interface & Implementation
- **Distinguish between declarations and definitions.** Declarations are used as interfaces, definitions as implementations.
- **Prefer modules over headers** where modules are supported.
- **Header Files:**
    - Use header files to represent interfaces and to emphasize logical structure.
    - `#include` a header in the source file that implements its functions.
    - Avoid non-inline function definitions in headers.

### Namespaces
- **Use namespaces** to express logical structure.
- **Using-directives:**
    - Use using-directives for transition, for foundational libraries (such as `std`), or within a local scope.
    - **Don't put a using-directive in a header file.**

### Function Arguments & Return Values
- **Argument Passing:**
    - Pass "small" values by value and "large" values by reference.
    - Prefer pass-by-const reference over plain pass-by-reference.
- **Return Values:**
    - Return values as function-return values (rather than by out-parameters).
    - **Don't overuse return-type deduction.**
    - **Don't overuse structured binding;** a named return type often gives more readable code.

## 4. Error Handling

### Error Handling Strategy
- **Throw an exception** to indicate that you cannot perform an assigned task.
- **Use exceptions for error handling only.**
- **Expected events:** Failing to open a file or to reach the end of an iteration are expected events and not exceptional.
- **Use error codes** when an immediate caller is expected to handle the error.
- **Throw an exception** for errors expected to percolate up through many function calls.
- **Prefer exceptions:** If in doubt whether to use an exception or an error code, prefer exceptions.
- **Early design:** Develop an error-handling strategy early in a design.

### Exceptions Implementation
- **User-defined types:** Use purpose-designed user-defined types as exceptions (not built-in types).
- **Don't catch everything:** Don't try to catch every exception in every function.
- **Standard hierarchy:** You don't have to use the standard-library exception class hierarchy.
- **RAII:** Prefer RAII to explicit try-blocks.
- **`noexcept`:**
    - If your function may not throw, declare it `noexcept`.
    - Don't apply `noexcept` thoughtlessly.

### Invariants & Assertions
- **Constructors:** Let a constructor establish an invariant, and throw if it cannot.
- **Design around invariants:** Design your error-handling strategy around invariants.
- **Compile-time checks:** What can be checked at compile time is usually best checked at compile time.
- **Assertions:** Use an assertion mechanism to provide a single point of control of the meaning of failure.
- **Concepts:** Concepts are compile-time predicates and therefore often useful in assertions.

## 5. Classes

### Concrete Types
- **Express ideas directly in code.**
- **Concrete types:** A concrete type is the simplest kind of class. Where applicable, prefer a concrete type over more complicated classes and over plain data structures.
- **Simple concepts:** Use concrete classes to represent simple concepts.
- **Performance:** Prefer concrete classes over class hierarchies for performance-critical components.
- **Constructors:** Define constructors to handle initialization of objects.
- **Containers:** If a class is a container, give it an initializer-list constructor.

### Class Design & Operators
- **Member functions:** Make a function a member only if it needs direct access to the representation of a class.
- **Operators:** Define operators primarily to mimic conventional usage.
- **Symmetric operators:** Use nonmember functions for symmetric operators.
- **Const correctness:** Declare a member function that does not modify the state of its object `const`.

### Resource Management
- **Destructors:** If a constructor acquires a resource, its class needs a destructor to release the source.
- **Scoping:** Avoid "naked" `new` and `delete` operations.
- **RAII:** Use resource handles and RAII to manage resources.
- **Smart pointers:** Use `unique_ptr` or `shared_ptr` to avoid forgetting to delete objects created using `new`.

### Abstract Types & Hierarchy
- **Abstract classes:** Use abstract classes as interfaces when complete separation of interface and implementation is needed.
- **Polymorphism:** Access polymorphic objects through pointers and references.
- **Constructors:** An abstract class typically doesn't need a constructor.
- **Class hierarchies:** Use class hierarchies to represent concepts with inherent hierarchical structure.
- **Virtual destructors:** A class with a virtual function should have a virtual destructor.
- **Overriding:** Use `override` to make overriding explicit in large class hierarchies.
- **Inheritance:** When designing a class hierarchy, distinguish between implementation inheritance and interface inheritance.

### Casting & Type Safety
- **Dynamic casting:** Use `dynamic_cast` where class hierarchy navigation is unavoidable.
- **References:** Use `dynamic_cast` to a reference type when failure to find the required class is considered a failure.
- **Pointers:** Use `dynamic_cast` to a pointer type when failure to find the required class is considered a valid alternative.

## 6. Essential Operations

### Object Lifecycle
- **Essential operations:** Control construction, copy, move, and destruction of objects.
- **Matched set:** Design constructors, assignments, and the destructor as a matched set of operations.
- **Completeness:** Define all essential operations or none.
- **Compiler generation:** If a default constructor, assignment, or destructor is appropriate, let the compiler generate it.
- **Pointer members:** If a class has a pointer member, consider if it needs a user-defined or deleted destructor, copy, and move.
- **User-defined destructor:** If a class has a user-defined destructor, it probably needs user-defined or deleted copy and move.
- **Explicit constructors:** By default, declare single-argument constructors `explicit`.
- **Resource handles:** If a class is a resource handle, it needs a user-defined constructor, a destructor, and non-default copy operations.

### Initialization & Defaults
- **Member initialization:** If a class member has a reasonable default value, provide it as a data member initializer.
- **Copying restrictions:** Redefine or prohibit copying if the default is not appropriate for a type.

### Resource Management
- **Resource safety:** Provide strong resource safety; that is, never leak anything that you think of as a resource.
- **RAII:** Manage all resources—memory and non-memory—using RAII.

### Operators & Conventions
- **Conventional usage:** Overload operators to mimic conventional usage.
- **Operator groups:** If you overload an operator, define all operations that conventionally work together.
- **Comparison operators:** If you define `<=>` for a type as non-default, also define `==`.

### Efficiency & Containers
- **Return by value:** Return containers by value (relying on copy elision and move for efficiency).
- **Copying:** Avoid explicit use of `std::copy()`.
- **Large operands:** For large operands, use `const` reference argument types.
- **Container design:** Follow the standard-library container design.

## 7. Templates

### General
- **Algorithms:** Use templates to express algorithms that apply to many argument types.
- **Containers:** Use templates to express containers.
- **Abstraction:** Use templates to raise the level of abstraction of code.
- **Type safety:** Templates are type safe, but for unconstrained templates checking happens too late.
- **Deduction:** Let constructors or function templates deduce class template argument types.

### Function Objects & Lambdas
- **Function objects:** Use function objects as arguments to algorithms.
- **Lambdas:** Use lambda if you need a simple function object in one place only.
- **Virtual functions:** A virtual function member cannot be a template member function.

### Advanced Mechanisms
- **RAII:** Use `finally()` to provide RAII for types without destructors that require "cleanup operation".
- **Aliases:** Use template aliases to simplify notation and hide implementation details.
- **Compile-time selection:** Use `if constexpr` to provide alternative implementations without run-time overhead.

## 8. Concepts & Generic Programming

### General
- **Compile-time programming:** Templates provide a general mechanism for compile-time programming.
- **Initial implementation:** When designing a template, use a concrete version for initial implementation, debugging, and measurement.
- **Containers & Ranges:** Use templates to express containers and ranges.
- **Variadics:** Use variadic templates when you need a function that takes a variable number of arguments of a variety of types.
- **Lambdas:** Use a lambda if you need a simple function object in one place only.
- **Duck-typing:** Unconstrained templates offer compile-time "duck-typing".

### Concepts Design
- **Concept requirements:** When designing a template, carefully consider the concepts (requirements) assumed for its template arguments.
- **Design tool:** Use concepts as a design tool.
- **Explicit concepts:** Specify concepts for all template arguments.
- **Named concepts:** Whenever possible use named concepts (e.g., standard-library concepts).
- **Meaningful semantics:** Avoid "concepts" without meaningful semantics.
- **Operations:** Require a complete set of operations for a concept.
- **Complex requirements:** Avoid `requires requires`.
- **Auto:** `auto` is the least constrained concept.

### Code Organization
- **Header definitions:** When using header files, `#include` template definitions (not just declarations) in every translation unit that uses them.
- **Scope:** To use a template, make sure its definition (not just its declaration) is in scope.

## 9. Library Overview

### General Philosophy
- **Don't reinvent the wheel:** Use libraries.
- **Standard library preference:** When you have a choice, prefer the standard library over other libraries.
- **Standard library limitations:** Do not think that the standard library is ideal for everything.
- **Includes:** If you don't use modules, remember to `#include` the appropriate headers.
- **Namespace `std`:** Remember that standard-library facilities are defined in namespace `std`.
- **Ranges:** When using ranges, remember to explicitly qualify algorithm names.
- **Modules:** Prefer importing modules over `#include`ing header files.

## 10. Strings & Regular Expressions

### Strings
- **Ownership:** Use `std::string` to own character sequences.
- **String operations:** Prefer string operations to C-style string functions.
- **Usage:** Use `string` to declare variables and members rather than as a base class.
- **Return by value:** Return strings by value (rely on move semantics and copy elision).
- **Substrings:** Directly or indirectly, use `substr()` to read substrings and `replace()` to write substrings.
- **Sizing:** A `string` can grow and shrink, as needed.
- **Range checking:** Use `at()` rather than iterators or `[]` when you want range checking.
- **Optimization:** Use iterators and `[]` rather than `at()` when you want to optimize speed.
- **Loops:** Use a range-for to safely minimize range checking.
- **Input:** `string` input doesn't overflow.
- **C-style strings:** Use `c_str()` or `data()` to produce a c-style string representation of a `string` (only) when you have to.
- **Numeric conversion:** Use a `stringstream` or a generic value extraction function (such as `to<X>`) for numeric conversion of strings.
- **`basic_string`:** A `basic_string` can be used to make strings of characters on any type.
- **Suffixes:** Use the `s` suffix for string literals meant to be standard-library strings.

### `string_view`
- **`string_view` usage:** Use `string_view` as an argument of functions that need to read character sequences stored in various ways.
- **`string_view` mental model:** Think of a `string_view` as a kind of pointer with a size attached; it does not own its characters.
- **Suffixes:** Use the `sv` suffix for string literals meant to be standard-library `string_views`.

### Regular Expressions
- **Matching:** Use `regex_match()` to match a complete input.
- **Searching:** Use `regex_search()` to search for a pattern in an input stream.
- **Standards:** The regular expression notation can be adjusted to match various standards.
- **Default notation:** The default regular expression notation is that of ECMAScript.
- **Restraint:** Be restrained; regular expressions can easily become a write-only language.
- **Subpatterns:** Note that `\i` for a digit `i` allows you to express a subpattern in terms of a previous subpattern.
- **Laziness:** Use `?` to make patterns "lazy".
- **Iterators:** Use `regex_iterators` for iterating over a stream looking for a pattern.

## 11. Input and Output

### General
- **Type safety:** iostreams are type-safe, type-sensitive, and extensible.
- **Character-level input:** Use character-level input only when you have to.
- **Ill-formed input:** When reading, always consider ill-formed input.
- **Avoid `endl`:** Avoid `endl` (if you don't know what `endl` is, you haven't missed anything).
- **User-defined types:** Define `<<` and `>>` for user-defined types with values that have meaningful textual representations.

### Standard Streams
- **Output streams:** Use `cout` for normal output and `cerr` for errors.
- **Character types:** There are iostreams for ordinary characters and wide characters, and you can define an iostream for any kind of character.
- **Binary I/O:** Binary I/O is supported.
- **Stream types:** There are standard iostreams for standard I/O streams, files, and strings.

### Input & Output Operations
- **Chaining output:** Chain `<<` operations for a terser notation.
- **Chaining input:** Chain `>>` operations for a terser notation.
- **String input:** Input into strings does not overflow.
- **Whitespace:** By default `>>` skips initial whitespace.
- **Error handling:** Use the stream state `fail` to handle potentially recoverable I/O errors.
- **Custom operators:** We can define `<<` and `>>` operators for our own types.
- **Extensibility:** We don't need to modify `istream` or `ostream` to add new `<<` and `>>` operators.

### Formatting
- **Manipulators:** Use manipulators or `format()` to control formatting.
- **Precision:** `precision()` specifications apply to all following floating-point output operations.
- **Float format:** Floating-point format specifications (e.g., `scientific`) apply to all following floating-point output operations.
- **Headers:** `#include <ios>` or `<iostream>` when using standard manipulators.
- **Sticky manipulators:** Stream formatting manipulators are "sticky" for use for many values in a stream.
- **Argument manipulators:** `#include <iomanip>` when using standard manipulators taking arguments.
- **Date & Time:** We can output time, dates, etc. in standard formats.

### Streams & Files
- **Copying:** Don't try to copy a stream: streams are move only.
- **File streams:** Remember to check that a file stream is attached to a file before using it.
- **In-memory formatting:** Use stringstreams or memory streams for in-memory formatting.
- **Conversions:** We can define conversions between any two types that both have string representation.

### C-Style I/O & Filesystem
- **C-style I/O:** C-style I/O is not type-safe.
- **Sync:** Unless you use `printf`-family functions, call `ios_base::sync_with_stdio(false)`.
- **Filesystem:** Prefer `<filesystem>` to direct use of platform-specific interfaces.


## 12. Containers

### General
- **Sequences:** An STL container defines a sequence.
- **Resource handles:** STL containers are resource handles.
- **Default container:** Use `vector` as your default container.
- **Traversals:** For simple traversals of a container, use a range-for loop or a `begin`/`end` pair of iterators.

### Sizing & Capacity
- **Reserve:** Use `reserve()` to avoid invalidating pointers and iterators to elements.
- **Reserve performance:** Don't assume performance benefits from `reserve()` without measurement.
- **Growth:** Use `push_back()` or `resize()` on a container rather than `realloc()` on an array.
- **Resized iterators:** Don't use iterators into a resized `vector`.

### Element Access & Safety
- **Range checking:** Do not assume that `[]` range checks.
- **Guaranteed range checks:** Use `at()` when you need guaranteed range checks.
- **Range errors:** Use range-for and standard-library algorithms for cost-free avoidance of range errors.

### Element Semantics
- **Copy semantics:** Elements are copied into a container.
- **Polymorphism:** To preserve polymorphic behavior of elements, store pointers (built-in or user-defined).
- **Insertion efficiency:** Insertion operations, such as `insert()` and `push_back()`, are often surprisingly efficient on a `vector`.

### Container Selection
- **Forward list:** Use `forward_list` for sequences that are usually empty.
- **Performance:** When it comes to performance, don't trust your intuition: measure.
- **Map implementation:** A `map` is usually implemented as a red-black tree.
- **Unordered map:** An `unordered_map` is a hash table.
- **Compact data:** Prefer compact and contiguous data structures.
- **List cost:** A `list` is relatively expensive to traverse.
- **Fast lookup:** Use unordered containers if you need fast lookup for large amounts of data.
- **Ordered iteration:** Use ordered containers (e.g., `map` and `set`) if you need to iterate over their elements in order.
- **No natural order:** Use unordered containers (e.g., `unordered_map`) for element types with no natural order (i.e., no reasonable `<`).
- **Stable pointers:** Use associative containers (e.g., `map` and `list`) when you need pointers to elements to be stable as the size of the container changes.

### Passing & Initialization
- **Passing containers:** Pass a container by reference and return a container by value.
- **Initializer syntax:** For a container, use the `()`-initializer syntax for sizes and the `{}`-initializer syntax for sequences of elements.

### Hashing
- **Hash function:** Experiment to check that you have an acceptable hash function.
- **Combining hashes:** A hash function obtained by combining standard hash functions for elements using the exclusive-or operator (`^`) is often good.

### Best Practices
- **Standard containers:** Know your standard-library containers and prefer them to handcrafted data structures.
- **Memory performance:** If your application is suffering performance problems related to memory, minimize free-store use and/or consider using a specialized allocator.

## 13. Algorithms

### General
- **Sequences:** An STL algorithm operates on one or more sequences.
- **Input sequences:** An input sequence is half-open and defined by a pair of iterators.
- **Custom iterators:** You can define your own iterators to serve special needs.
- **I/O streams:** Many algorithms can be applied to I/O streams.
- **Searching:** When searching, an algorithm usually returns the end of the input sequence to indicate "not found".
- **Non-modifying sequences:** Algorithms do not directly add or subtract elements from their argument sequences.
- **Loop replacement:** When writing a loop, consider whether it could be expressed as a general algorithm.

### Best Practices
- **Type aliases:** Use using-type aliases to clean up messy notation.
- **Function objects:** Use predicates and other function objects to give standard algorithms a wider range of meanings.
- **Predicates:** A predicate must not modify its argument.
- **Standard algorithms:** Know your standard-library algorithms and prefer them to hand-crafted loops.

## 14. Ranges

### General
- **Range algorithms:** When the pair-of-iterators style becomes tedious, use a range algorithm.
- **Explicit names:** When using a range algorithm, remember to explicitly introduce its name.
- **Pipelines:** Pipelines of operations on a range can be expressed using views, generators, and filters.
- **Sentinels:** To end a range with a predicate, you need to define a sentinel.
- **Static assertions:** Using `static_assert`, we can check that a specific type meets the requirements of a concept.
- **Custom algorithms:** If you want a range algorithm and there isn't one in the standard, just write your own.

### Best Practices
- **Regular types:** The ideal for types is regular.
- **Standard concepts:** Prefer standard-library concepts where they apply.
- **Parallel execution:** When requesting parallel execution, be sure to avoid data races and deadlock.