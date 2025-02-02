# 1 Data Types and Quality

<br>

## Numerical / Quantitative
- Variables that are measured are **Numerical** variables.
- Numerical variables are a combination of the measurement and the unit.
    - Without the unit, a numerical variable is just a number.
- Numerical variables can be further divided into two types:
    1. **Discrete** variables
        - Counting gives us whole numbers and discrete variables.
        - `Example: Number of students in a class`
    2. **Continuous** variables
        - Measuring gives us potentially partial values and continuous variables.
        - `Example: Height of students in a class`

<br>

## Categorical / Qualitative
- Variables that are categorized are **Categorical** variables.
- Categorical variables describe characteristics with words or relative values.
- Categorical variables can be further divided into three types:
    1. **Ordinal** variables
        - also called *likert scale*
        - Variables that have a natural order.
        - `Example: Grades (A, B, C, D, F)`
        - `Example: Sizes (Small, Medium, Large)`
    2. **Nominal** variables
        - Variables that do not have a natural order.
        - `Example: Colors (Red, Blue, Green)`
    3. **Binary / Dichotomous** variables
        - Variables that have only two categories.
        - `Example: "yes/no", "true/false", "0/1"`

<br>

## Messy Data
- Messy data is data that violates one of the tidy datasetrules 
    1. Each variable forms a column
    2. Eachobservation forms a row 
    3. Each type of observationalunit forms a table
- **Typos** like `Tuuullip` instead of `Tulip`
    - Might be divided into two categories: `Tuuullip` and `Tulip`
    - Can make grouping similar terms difficult
- **Missing data** like `NaN` or `NULL`
    - inaccurate summary statistics
    - `Example: Average of [1, 2, 3, NaN, 5] = 2.75`
- **Inconsistent coding** like `Male`, `male`, `M`, `m` in the same column
    - Can result in splitting categories
    - might result into errors in analysis, if we want to group them 

<br>

## Types of Missing Data
- **Missing Completely at Random (MCAR)**
    - Randomness missing without systemantic reason
    - No deeper meaning to why the data is missing
    - `Example: Human error for data entry`
- **Missing at Random (MAR)**
    - Missing explained by other variables
    - Not really "random"
    - `Example: Missing of Tree Hight only for the trees in the shadow`
- **Structurally Missing**
    - we wouldn’t expect a value there to begin with.
    - `Example: Asking for the number of children to a person who doesn't have any`
    - `Example: Collecting data about fruit of a tree that doesn't have any`

<br>

## Accuracy 
- How good is the data?
- How well records reflect reality?
- *Standardization* is essential for accuracy.
- It all comes to the question: 
    - *Are these measurements (or categorizations) correct?*
    - It requires a critical evaluation of our dataset
    1. **First**, thinking about the data against expectations and common sense is crucial for spotting issues with accuracy.
        - Inspecting the distribution
        - Inspecting the outliers
    2. **Second**, critically considering how error could have crept in during the data collection process will help you group and evaluate the data to uncover systematic inconsistencies.
    3. **Finally**,  identifying ways that duplicate values could have been created goes a long way towards ensuring that reality is only represented once in your data. 
        - Distinguish between what was human collected versus programmatically generated and using that distinction to segment the data

<br>

## Validity
- We have to make sure that our data actually measures what we think it is measuring
- Do I have data that aligns to my question?
- Special kind of quality measure
    - It’s about the relationship between the dataset and its purpose
    - *A dataset can be valid for one question and invalid for another*

<br>

## Representive Samples
- Goal of a sample is to represent a population
- Any time a sample is made that does NOT reflect the entire population, it is a **sampling error**.
- **Representive Sample** 
    - Should look like the population in as many characteristics as possible.
    - Should be large enough to be representative.
    - Needs to include many different kind of people from many different backgrounds / places.
    - Good sample size ca. `10%` of the population
    - `Example: 10,000 plants with 25% tulips, 25% daffodils, 20% hyacinths, 20% hydrangeas, and 10% irises -> 1,000 plants with 25% tulips, 25% daffodils, 20% hyacinths, 20% hydrangeas, and 10% irises` 
- **Convenience Sample**
    - Sampling error
    - A sample that is easy to collect
    - Not good for representing broader population
    - `Example: Surveying people in the street`
- **Bias**
    - A systematic error in the data
    - `Example: Surveying people in the street in a wealthy neighborhood`