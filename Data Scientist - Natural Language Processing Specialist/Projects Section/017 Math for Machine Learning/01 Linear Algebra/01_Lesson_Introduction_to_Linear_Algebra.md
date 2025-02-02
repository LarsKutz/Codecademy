# Tasks: Introduction to Linear Algebra

<br>

## Content
- [Vectors](#vectors)
- [Basic Vector Operations](#basic-vector-operations)
- [Vector Dot Product](#vector-dot-product)

<br>

## Vectors
**Formulas**
- Magnitude of a Vector (or Length/Norm of a Vector):
$$ ||v|| = \sqrt{\sum_{i=1}^{n} v_i^2} $$

<br>

**Task**
1. Find the magnitudes of the following vectors:
$$ a = \begin{bmatrix} 3 \\ -4 \end{bmatrix} $$
$$ b = \begin{bmatrix} 0 \\ 26 \end{bmatrix} $$
$$ c = \begin{bmatrix} 26 \\ 0 \end{bmatrix} $$
$$ d = \begin{bmatrix} -12 \\ 13 \end{bmatrix} $$

<br>

**Solution**
$$ ||a|| = \sqrt{3^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = 5 $$
$$ ||b|| = \sqrt{0^2 + 26^2} = \sqrt{0 + 676} = \sqrt{676} = 26 $$
$$ ||c|| = \sqrt{26^2 + 0^2} = \sqrt{676 + 0} = \sqrt{676} = 26 $$
$$ ||d|| = \sqrt{(-12)^2 + 13^2} = \sqrt{144 + 169} = \sqrt{313} \approx 17.69 $$

<br>

## Basic Vector Operations
**Formulas**
- Scalar Multiplication: 
$$ k \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} kx \\ ky \\ kz \end{bmatrix} $$
- Vector Addition and Subtraction:
$$ \begin{bmatrix} x_1 \\ y_1 \\ z_1 \end{bmatrix} + 2 \begin{bmatrix} x_2 \\ y_2 \\ z_2 \end{bmatrix} - 3 \begin{bmatrix} x_3 \\ y_3 \\ z_3 \end{bmatrix} = \begin{bmatrix} x_1 + 2x_2 - 3x_3 \\ y_1 + 2y_2 - 3y_3 \\ z_1 + 2z_2 - 3z_3 \end{bmatrix} $$

<br>

**Task**
1. $$ 3 \begin{bmatrix} 2 \\ -7 \\ 1 \end{bmatrix} $$
2. $$ \begin{bmatrix} 10 \\ -7 \end{bmatrix} + \begin{bmatrix} -14 \\ 7 \end{bmatrix} $$
3. $$ \begin{bmatrix} 14 \\ -3 \\ 2 \end{bmatrix} + 7 \begin{bmatrix} -1 \\ 3 \\ -5 \end{bmatrix} - 5 \begin{bmatrix} 6 \\ -4 \\ -1 \end{bmatrix} $$

<br>

**Solution**
1. $$ 3 \begin{bmatrix} 2 \\ -7 \\ 1 \end{bmatrix} = \begin{bmatrix} 3(2) \\ 3(-7) \\ 3(1) \end{bmatrix} = \begin{bmatrix} 6 \\ -21 \\ 3 \end{bmatrix} $$
2. $$ \begin{bmatrix} 10 \\ -7 \end{bmatrix} + \begin{bmatrix} -14 \\ 7 \end{bmatrix} = \begin{bmatrix} 10 + (-14) \\ -7 + 7 \end{bmatrix} = \begin{bmatrix} -4 \\ 0 \end{bmatrix} $$
3. $$ \begin{bmatrix} 14 \\ -3 \\ 2 \end{bmatrix} + 7 \begin{bmatrix} -1 \\ 3 \\ -5 \end{bmatrix} - 5 \begin{bmatrix} 6 \\ -4 \\ -1 \end{bmatrix} = \begin{bmatrix} 14 + 7(-1) - 5(6) \\ -3 + 7(3) - 5(-4) \\ 2 + 7(-5) - 5(-1) \end{bmatrix} = \begin{bmatrix} 14 - 7 - 30 \\ -3 + 21 + 20 \\ 2 - 35 + 5 \end{bmatrix} = \begin{bmatrix} -23 \\ 38 \\ -28 \end{bmatrix} $$

<br>

## Vector Dot Product
**Formulas**
$$ a \cdot b = \sum_{i=1}^{n} a_i  b_i $$
- Angle between two vectors:
$$ \cos(\theta) = \frac{a \cdot b}{||a||||b||} $$

<br>

**Task**
$$ a = \begin{bmatrix} -17 \\ 22 \end{bmatrix}, b = \begin{bmatrix} 0 \\ 32 \end{bmatrix} $$

<br>

**Solution**
- Dot Product:
$$ a \cdot b = (-17)(0) + 22(32) = 0 + 704 = 704 $$
- Angle between two vectors:
    - Magnitude of a:
    $$ ||a|| = \sqrt{(-17)^2 + 22^2} = \sqrt{289 + 484} = \sqrt{773} \approx 27.78 $$
    - Magnitude of b:
    $$ ||b|| = \sqrt{0^2 + 32^2} = \sqrt{0 + 1024} = \sqrt{1024} = 32 $$
    - Angle:
    $$ \theta = \arccos \frac{704}{27.8 \cdot 32} = \arccos \frac{704}{889.6} \approx 37.69^{\circ} $$

<br>

## Matrices