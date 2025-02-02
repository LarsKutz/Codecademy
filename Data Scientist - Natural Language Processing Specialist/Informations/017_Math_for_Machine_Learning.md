# 17. Math for Machine Learning

<br>

## Content
- **Linear Algebra**
    - **Introduction to Linear Algebra**
        - [Vectors](#vectors)
        - [Basic Vector Operations](#basic-vector-operations)
            - [Scalar Multiplication](#scalar-multiplication)
            - [Vector Addition and Subtraction](#vector-addition-and-subtraction)
        - [Vector Dot Product](#vector-dot-product)
        - [Matrices](#matrices)
        - [Matrix Operations](#matrix-operations)
        - [Special Matrices](#special-matrices)
        - [Linear Systems in Matrix Form](#linear-systems-in-matrix-form)
        - [Gauss-Jordan Elimination](#gauss-jordan-elimination)
        - [Inverse Matrices](#inverse-matrices)
    - **Linear Algebra with Python**
        - [Using NumPy Arrays](#using-numpy-arrays)
        - [Using Numpy for Linear Algebra Operations](#using-numpy-for-linear-algebra-operations)
        - [Special Matrices](#special-matrices)
        - [Additional Linear Algebra Operations](#additional-linear-algebra-operations)

<br>

## Vectors
- The fundamental building blocks of linear algebra are *vectors*. 
- Vectors are defined as quantities having both direction and magnitude, compared to *scalar* quantities that only have magnitude. 
- In order to have direction and magnitude, vector quantities consist of two or more elements of data. 
- The *dimensionality* of a vector is determined by the number of numerical elements in that vector. 
- For example, a vector with four elements would have a dimensionality of four.

<br>

- Let’s take a look at examples of a scalar versus a vector. 
- A car driving at a speed of 40mph is a scalar quantity. 
- Describing the car driving 40mph to the east would represent a two-dimensional vector quantity since it has a magnitude in both the x and y directions.

<br>

- Vectors can be represented as a series of numbers enclosed in parentheses, angle brackets, or square brackets. 
- In this lesson, we will use square brackets for consistency. 
- For example, a three-dimensional vector is written as:  
$$v = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$
- The magnitude (or length) of a vector, $||v||$, can be calculated with the following formula:  
$$||v|| = \sqrt{\sum_{i=1}^{n} v_i^2}$$
- This formulates translates to the sum of each vector component squared, which can be also written out as:  
$$||v|| = \sqrt{v_1^2 + v_2^2 + v_3^2 + \ldots + v_n^2}$$
- **Let’s look at an example**: 
    - We are told that a ball is traveling through the air and given the velocities of that ball in its $x$, $y$, and $z$ directions in a standard Cartesian coordinate system. 
    - The velocity component values are:
        - $v_x = -12$
        - $v_y = 8$
        - $v_z = -2$
    - Convert the velocities into a vector, and find the total speed of the ball. 
    - (Hint: the speed of the ball is the magnitude of the velocity vector!)  
$$v = \begin{bmatrix} -12 \\ 8 \\ -2 \end{bmatrix}$$
$$ ||v|| = \sqrt{(-12)^2 + 8^2 + (-2)^2} = \sqrt{144 + 64 + 4} = \sqrt{212} \approx 14.56$$

<br>

## Basic Vector Operations
- Now that we know what a vector is and how to represent one, we can begin to perform various operations on vectors and between different vectors. 
- As we’ve previously discussed, the basis of linear algebra is the linear combinations we can generate between vectors (and matrices).

<br>

### Scalar Multiplication
- Any vector can be multiplied by a scalar, which results in every element of that vector being multiplied by that scalar individually.  
$$k \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} kx \\ ky \\ kz \end{bmatrix}$$
- Multiplying vectors by scalars is an *associative operation*, meaning that rearranging the parentheses in the expression does not change the result. 
- For example, we can say ***2(a3) = (2a)3***.

<br>

### Vector Addition and Subtraction
- Vectors can be added and subtracted from each other when they are of the same dimension (same number of components). 
- Doing so adds or subtracts corresponding elements, resulting in a new vector of the same dimension as the two being summed or subtracted. 
- Below is an example of three-dimensional vectors being added and subtracted together.  
$$\begin{bmatrix} x_1 \\ y_1 \\ z_1 \end{bmatrix} + 2 \begin{bmatrix} x_2 \\ y_2 \\ z_2 \end{bmatrix} - 3 \begin{bmatrix} x_3 \\ y_3 \\ z_3 \end{bmatrix} = \begin{bmatrix} x_1 + 2x_2 - 3x_3 \\ y_1 + 2y_2 - 3y_3 \\ z_1 + 2z_2 - 3z_3 \end{bmatrix}$$
- Vector addition is *commutative*, meaning the order of the terms does not matter. 
- For example, we can say ***(a+b = b+a)***. 
- Vector addition is also associative, meaning that ***(a + (b+c) = (a+b) + c)***.

<br>

## Vector Dot Product
- An important vector operation in linear algebra is the dot product. 
- A *dot product* takes two equal dimension vectors and returns a single scalar value by summing the products of the vectors’ corresponding components. This can be written out formulaically as:  
$$a \cdot b = \sum_{i=1}^{n} a_i  b_i$$
- The dot product operation is both commutative ***(a · b = b · a)*** and distributive ***(a · (b+c) = a · b + a · c)***.
- **The resulting scalar value represents how much one vector “goes into” the other vector.** 
- If two vectors are perpendicular (or orthogonal), their dot product is equal to 0, as neither vector “goes into the other.”
- Let’s take a look at an example dot product. Consider the following two vectors:  
$$a = \begin{bmatrix} 3 \\ 2 \\ -3 \end{bmatrix} \quad b = \begin{bmatrix} 0 \\ -3 \\ -6 \end{bmatrix}$$
- To find the dot product between these two vectors, we do the following:  
$$a \cdot b = 3 \cdot 0 + 2 \cdot (-3) + (-3) \cdot (-6) = 0 - 6 + 18 = 12$$
- The dot product can also be used to find the magnitude of a vector and the angle between two vectors. 
- To find the magnitude, we can reference Exercise 2 and see that the magnitude of a vector is simply the square root of a vector’s dot product with itself.  
$$||a|| = \sqrt{a \cdot a}$$
- To find the angle between two vectors, we rely on the dot product between the two vectors and use the following equation.  
$$\cos(\theta) = \frac{a \cdot b}{||a||||b||}$$
- Let’s look at the same two vectors from above:  
$$a = \begin{bmatrix} 3 \\ 2 \\ -3 \end{bmatrix} \quad b = \begin{bmatrix} 0 \\ -3 \\ -6 \end{bmatrix}$$
- To find the angle between these two vectors, we do the following:  
$$\theta = \arccos \frac{3 \cdot 0 + 2 \cdot (-3) + (-3) \cdot (-6)}{\sqrt{3^2 + 2^2 + (-3)^2} \cdot \sqrt{0^2 + (-3)^2 + (-6)^2}}$$
- Solving this, we end up with:  
$$\theta = 67.58^{\circ}$$

<br>

## Matrices