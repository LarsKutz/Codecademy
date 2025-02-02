# 7 Python Pandas for Data Science

[10 min. User Guide](https://pandas.pydata.org/docs/user_guide/10min.html)

<br>

## Contents
- **Creating, Loading and Selecting DataFrames**
    - [Lambda Functions](#lambda-functions)
    - [Create a DataFrame I - Using a Dictionary](#create-a-dataframe-i---using-a-dictionary)
    - [Create a DataFrame II - Using a List](#create-a-dataframe-ii---using-a-list)
    - [Create a DataFrame III - Loading and Saving CSVs](#create-a-dataframe-iii---loading-and-saving-csvs)
    - [Inspect a DataFrame](#inspect-a-dataframe)
    - [Select Columns](#select-columns)
    - [Selecting Multiple Columns](#selecting-multiple-columns)
    - [Select Rows](#select-rows)
    - [Selecting Multiple Rows](#selecting-multiple-rows)
    - [Select Rows with Logic I](#select-rows-with-logic-i)
    - [Select Rows with Logic II](#select-rows-with-logic-ii)
    - [Select Rows with Logic III](#select-rows-with-logic-iii)
    - [Setting Indices](#setting-indices)
- **Modifying DataFrames**
    - [Adding a Column I](#adding-a-column-i)
    - [Adding a Column II](#adding-a-column-ii)
    - [Adding a Column III](#adding-a-column-iii)
    - [Performing Column Operations](#performing-column-operations)
    - [Applying a Lambda to a Column](#applying-a-lambda-to-a-column)
    - [Apllying a Lambda to a Row](#apllying-a-lambda-to-a-row)
    - [Renaming Columns I](#renaming-columns-i)
    - [Rename Columns II](#rename-columns-ii)
- **Aggregates in Pandas**
    - [Calculating Column Statistics](#calculating-column-statistics)
    - [Calculating Aggregate Functions I](#calculating-aggregate-functions-i)
    - [Calculating Aggregate Functions II](#calculating-aggregate-functions-ii)
    - [Calculating Aggregate Functions III](#calculating-aggregate-functions-iii)
    - [Calculating Aggregate Functions IV](#calculating-aggregate-functions-iv)
    - [Pivot Tables](#pivot-tables)
- **Working with Multiple DataFrames**
    - [Inner Merge](#inner-merge)
    - [Merge on Specific Columns I](#merge-on-specific-columns-i)
    - [Merge on Specific Columns II](#merge-on-specific-columns-ii)
    - [Outer Merge](#outer-merge)
    - [Left and Right Merge](#left-and-right-merge)
    - [Concatenate DataFrames](#concatenate-dataframes)

<br>

## Lambda Functions
-  Is a one-line shorthand for function. A simple lambda function might look like this:

```python
add_two = lambda my_input: my_input + 2

print(add_two(3))
print(add_two(100))
print(add_two(-2))

# Output
>>> 5
>>> 102
>>> 0
```

```python
is_substring = lambda my_string: my_string in "This is the master string"

print(is_substring('I'))
print(is_substring('am'))
print(is_substring('the'))
print(is_substring('master'))

# Output
>>> False
>>> False
>>> True
>>> True
```

### Lambda in Pandas
- Lambda functions are useful when you need a quick function to do some work for you.
- The Pandas `apply()` function can be used to apply a function on every value in a column or row of a DataFrame, and transform that column or row to the resulting values.
    - By default, it will apply a function to all values of a column.
    - To perform it on a row instead, you can specify the argument `axis=1` in the `apply()` function call.

```python
# This function doubles the input value
def double(x):
  return 2*x

# Apply this function to double every value in a specified column
df.column1 = df.column1.apply(double)

# Lambda functions can also be supplied to `apply()`
df.column2 = df.column2.apply(lambda x : 3*x)

# Applying to a row requires it to be called on the entire DataFrame
df['newColumn'] = df.apply(lambda row: row['column1'] * 1.5 + row['column2'], axis=1)
```

<br>

## Create a DataFrame I - Using a Dictionary
- A DataFrame is an object that stores data as rows and columns.
- You can think of a DataFrame as a spreadsheet or as a SQL table.
- DataFrames have rows and columns.
    - Each column has a name, which is a string.
    -  Each row has an index, which is an integer.
    - DataFrames can contain many different data types: strings, ints, floats, tuples, etc.

```py
# Creating DataFrame with a dictionary

df1 = pd.DataFrame({
    'name': ['John Smith', 'Jane Doe', 'Joe Schmo'],
    'address': ['123 Main St.', '456 Maple Ave.', '789 Broadway'],
    'age': [34, 28, 51]
})
```

<br>

## Create a DataFrame II - Using a List
- You can also add data using lists.

```py
# Creating DataFrame with a list

df2 = pd.DataFrame([
    ['John Smith', '123 Main St.', 34],
    ['Jane Doe', '456 Maple Ave.', 28],
    ['Joe Schmo', '789 Broadway', 51]
], columns=['name', 'address', 'age'])
```

<br>

## Create a DataFrame III - Loading and Saving CSVs
- You can also create a DataFrame from a CSV file.

```py
# Creating DataFrame with a CSV file

df3 = pd.read_csv('file_name.csv')
```

- You can also save a DataFrame to a CSV file.

```py
# Saving DataFrame to a CSV file

df3.to_csv('new_file_name.csv')
```

<br>

## Inspect a DataFrame
- When you create a new DataFrame, you can take a look at its contents by using the `.head()` method or the `.info()` method.
- The `.head()` method prints the first five rows of the DataFrame.
- The `.info()` method prints a summary statistics for each column.

```py
# Inspecting a DataFrame

df1.head()
df1.info()
```

- If the Dataframe is small, you can just print the entire thing.

```py
# Inspecting a DataFrame

print(df1)
```

<br>

## Select Columns
- You can select a single column from a DataFrame by using square brackets.
- You can also select a single column using dot notation.
    - doesn’t start with a number, doesn’t contain spaces or special characters, etc.
- You can also select multiple columns from a DataFrame by using double square brackets.

```py
# Selecting a single column

df1['name']
df1.name
```

<br>

## Selecting Multiple Columns
- You can select multiple columns from a DataFrame by using double square brackets.

```py
# Selecting multiple columns

df1[['name', 'address']]
```

<br>

## Select Rows
- DataFrames are zero-indexed, meaning that we start with the 0th row and count up from there.
- When we select a single row, the result is a *Series* (just like when we select a single column).

```py
df.iloc[index_number]
```

## Selecting Multiple Rows
- `orders.iloc[3:7]` would select all rows starting at the 3rd row and up to but not including the 7th row (i.e., the 3rd row, 4th row, 5th row, and 6th row)
- `orders.iloc[:4]` would select all rows up to, but not including the 4th row (i.e., the 0th, 1st, 2nd, and 3rd rows)
- `orders.iloc[-3:]` would select the rows starting at the 3rd to last row and up to and including the final row

```py
# Selecting multiple rows

df1.iloc[1:3]
```

<br>

## Select Rows with Logic I
- You can select a subset of a DataFrame by using logical statements.

```py
df[df.MyColumnName == desired_column_value]

df[df.age == 30]

df[df.age > 30]

df[df.age < 30]

df[df.name != 'Clara Oswald']
```

<br>

## Select Rows with Logic II
- You can also combine multiple logical statements, as long as each statement is in parentheses.

```py
# | is the symbol for "or" and & is the symbol for "and"

df[(df.age < 30) | (df.name == 'Martha Jones')]
```

<br>

## Select Rows with Logic III
- Suppose we want to select the rows where the customer’s name is either “Martha Jones”, “Rose Tyler” or “Amy Pond”.

```py
df[df.name.isin(['Martha Jones', 'Rose Tyler', 'Amy Pond'])]
```

<br>

## Setting Indices
- When we select a subset of a DataFrame using logic, we end up with non-consecutive indices.
- This is inelegant and makes it hard to use `.iloc()`.
- We can fix this using the `.reset_index()` method.
    - Note that the old indices have been moved into a new column called `'index'`.  
    - Unless you need those values for something special, it’s probably better to use the keyword `drop=True` so that you don’t end up with that extra column.
- Using `.reset_index()` will return a new DataFrame, but we usually just want to modify our existing DataFrame.
    -  If we use the keyword `inplace=True` we can just modify our existing DataFrame.

```py
df.reset_index(drop=True, inplace=True)

df2 = df.reset_index(drop=True)
```

<br>

## Adding a Column I
- We might want to add new information or perform a calculation based on the data that we already have.
- One way that we can add a new column is by giving a list of the **same length as the existing DataFrame**.

```py
df['Quantity'] = [100, 150, 50, 35]
```

<br>

## Adding a Column II
- We can also add a new column that is the same for all rows in the DataFrame. 

```py
df['In Stock?'] = True

# Output
>>> 
           Product  Price  Quantity  In Stock?
0           Hammer     12       100      True
1            Nails      7       150      True
2  Screwdriver Set    45        50       True
3           Wrench    15        35       True
```

<br>

## Adding a Column III
- Finally, you can add a new column by performing a function on the existing columns.

```py
df['Sales Tax'] = df.Price * 0.075

df['Total'] = df.Price + df['Sales Tax']
```

<br>

## Performing Column Operations
- Often, the column that we want to add is related to existing columns, but requires a calculation more complex than multiplication or addition.
- We can use the `apply` function to apply a function to every value in a particular column. 

```py
df['Name'] = df.Name.apply(str.upper)
```

<br>

## Applying a Lambda to a Column
- In Pandas, we often use lambda functions to perform **complex** operations on columns.
- For example, suppose that we want to create a column containing the email provider for each email address

```py
# Current Table
>>> 
        Name                Email
0  Alice Doe  alice.doe@gmail.com
1  Bob Smith  bob.smith@yahoo.com


df['Email Provider'] = df.Email.apply(lambda x: x.split('@')[-1])

# Output
>>> 
        Name                Email   Email Provider
0  Alice Doe  alice.doe@gmail.com        gmail.com  
1  Bob Smith  bob.smith@yahoo.com        yahoo.com
```

<br>

## Apllying a Lambda to a Row
- We can also operate on multiple columns at once.
- If we use `apply` without specifying a single column and add the argument `axis=1`, the input to our lambda function **will be an entire row**, not a column. 
-  To access particular values of the row, we use the syntax row.`column_name` or `row[‘column_name’]`.

```py
# Current Table
>>>
0    Item	        Price	Is taxed?
1    Apple	        1.00	No
2    Milk	        4.20	No
3    Paper Towels	5.00	Yes
4    Light Bulbs	3.75	Yes

df['Price with Tax'] = df.apply(lambda row: row['Price'] * 1.075 if row['Is taxed?'] == 'Yes' else row['Price'], axis=1)

# Output
>>>
0    Item	        Price	Is taxed?	Price with Tax
1    Apple	        1.00	No	        1.00
2    Milk	        4.20	No	        4.20
3    Paper Towels	5.00	Yes	        5.38
4    Light Bulbs	3.75	Yes	        4.03
```

<br>

## Renaming Columns I
- When we get our data from other sources, we often want to change the column names. 

```py
df = pd.DataFrame({
    'name': ['John', 'Jane', 'Sue', 'Fred'],
    'age': [23, 29, 21, 18]
})
df.columns = ['First Name', 'Age']
```

- This command edits the **existing** DataFrame `df`.

<br>

## Rename Columns II
- You also can rename individual columns by using the `.rename` method.
- Pass a dictionary like the one below to the `columns` keyword argument:

```py
df = pd.DataFrame({
    'name': ['John', 'Jane', 'Sue', 'Fred'],
    'age': [23, 29, 21, 18]
})
df.rename(columns={'name': 'First Name', 'age': 'Age'}, inplace=True)
```

- Using `rename` with only the `columns` keyword will create a **new** DataFrame, leaving your original DataFrame unchanged.
- That’s why we also passed in the keyword argument `inplace=True`.
- Using `inplace=True` lets us edit the **original** DataFrame
- There are several reasons why .rename is preferable to `.columns`:
    - You can rename just one column
    - You can be specific about which column names are getting changed (with `.column` you can accidentally switch column names if you’re not careful)
- **Note**: If you misspell one of the original column names, this command won’t fail. It just won’t change anything.

<br>

## Calculating Column Statistics
- Aggregate functions summarize many data points (i.e., a column of a dataframe) into a smaller set of values.
- **Commands:**
    - `mean` Average of all values in column
    - `std`	Standard deviation
    - `median` Median
    - `max` Maximum value in column
    - `min` Minimum value in column
    - `count` Number of values in column
    - `nunique` Number of unique values in column
    - `unique` List of unique values in column
    - `value_counts` Number of times each unique value appears in column

```py	
# General command
df.column_name.command()

print(inventory.age)
>> [23, 25, 31, 35, 35, 46, 62]
print(inventory.age.median())
>> 35

print(inventory.state)
>> ['CA', 'CA', 'CA', 'CA', 'NY', 'NY', 'NJ', 'NJ', 'NJ', 'NJ', 'NJ', 'NJ', 'NJ']
print(inventory.state.nunique())
>> 3

print(inventory.color)
>> ['blue', 'blue', 'blue', 'blue', 'blue', 'green', 'green', 'orange', 'orange', 'orange']
print(inventory.color.unique())
>> ['blue', 'green', 'orange']
```

<br>

## Calculating Aggregate Functions I
- When we have a bunch of data, we often want to calculate aggregate statistics (mean, standard deviation, median, percentiles, etc.) over certain subsets of the data.

```py
# General command
df.groupby('column1').column2.measurement()

>>>
    student	assignment_name	grade
0   Amy	Assignment 1	75
1   Amy	Assignment 2	35
2   Bob	Assignment 1	99
3   Bob	Assignment 2	35
…		

grades = df.groupby('student').grade.mean() # create a new series

# Output
>>>
student
Amy    55.0
Bob    67.0
```

- `column1` is the column that we want to group by (`'student'` in our example)
- `column2` is the column that we want to perform a measurement on (`grade` in our example)
- `measurement` is the measurement function we want to apply (`mean` in our example)

<br>

## Calculating Aggregate Functions II
- After using `groupby`, we often need to clean our resulting data.
- As we saw in the previous exercise, the `groupby` function creates a new Series, not a DataFrame.
- Usually, we’d prefer that those indices were actually a column.
    -  In order to get that, we can use `reset_index()`.
    - This will transform our Series into a DataFrame and move the indices into their own column.

```py
# General command
df.groupby('column1').column2.measurement().reset_index()

id	tea	                category    caffeine	price
0	earl grey	           black	      38	    3
1	english breakfast	   black	      41	    3
2	irish breakfast	       black	      37	  2.5
3	jasmine	               green	      23	  4.5
4	matcha	               green	      48	    5
5	camomile	           herbal	       0	    3
…		

teas_counts = teas.groupby('category').id.count().reset_index()

# Output
>>>
    category	id
0	black	     3
1	green	     2
2	herbal	     1
```

<br>

## Calculating Aggregate Functions III
- Sometimes, the operation that you want to perform is more complicated than `mean` or `count`. 
- In those cases, you can use the `apply` method and lambda functions, just like we did for individual column operations. 
- **Note that the input to our lambda function will always be a list of values.**
- A great example of this is calculating percentiles. Suppose we have a DataFrame of employee information called `df` that has the following columns:
    - `id`: the employee’s id number
    - `name`: the employee’s name
    - `wage`: the employee’s hourly wage
    - `category`: the type of work that the employee does

```py
>>>
     id	         name	wage	category
0 10131	 Sarah Carney	  39	 product
1 14189	Heather Carey	  17	  design
2 15004	 Gary Mercado	  33   marketing
3 11204	   Cora Copaz	  27	  design
…

# np.percentile can calculate any percentile over an array of values
high_earners = df.groupby('category').wage.apply(lambda x: np.percentile(x, 75)).reset_index()

# Output
>>>
    category	wage
0	design	    27
1	marketing	33
2	product	    39
```

<br>

## Calculating Aggregate Functions IV
- Sometimes, we want to group by more than one column.
-  We can easily do this by passing a list of column names into the `groupby` method.

```py
>>>
      Location	    Date	Day of Week	Total Sales
0 West Village	February              1	    W	400
1 West Village	February              2	   Th	450
2      Chelsea	February              1	    W	375
3      Chelsea	February              2	   Th	390

df.groupby(['Location', 'Day of Week'])['Total Sales'].mean().reset_index()

# Output
>>>
      Location	Day of Week	Total Sales
0      Chelsea	         Th	        390
1      Chelsea	          W	        375
2 West Village	         Th	        450
3 West Village	          W	        400
```

<br>

## Pivot Tables
- When we perform a `groupby` across multiple columns, we often want to change how our data is stored.

```py
# General command
df.pivot(columns='ColumnToPivot', index='ColumnToBeRows', values='ColumnToBeValues')

# Current Table
>>>
Location	        Date	  Day of Week	Total Sales
West Village	February 1	            W	        400
West Village	February 2	           Th	        450
Chelsea	        February 1	            W	        375
Chelsea	        February 2	           Th	        390

unpivoted = df.groupby(['Location', 'Day of Week'])['Total Sales'].mean().reset_index()

# Output
>>>
      Location	Day of Week	Total Sales
0      Chelsea	         Th	        390
1      Chelsea	          W	        375
2 West Village	         Th	        450
3 West Village	          W	        400

pivoted = unpivoted.pivot(columns='Day of Week', index='Location', values='Total Sales').reset_index()

# Output
>>>
      Location	 Th	      W   
0      Chelsea	390	    375
1 West Village	450	    400
```

- Reorganizing a table in this way is called **pivoting**. The new table is called a **pivot table**.
- Just like with `groupby`, the output of a pivot command is a new DataFrame, but the indexing tends to be “weird”, so we usually follow up with `.reset_index()`.

<br>

## Inner Merge 
- Pandas can efficiently do this for the entire table. We use the `.merge()` method.
- The `.merge()` method looks for columns that are common between two DataFrames and then looks for rows where those column’s values are the same.
-  It then combines the matching rows into a single row in a new table.

```py
# Can merge only two DataFrames at a time
new_df = pd.merge(orders, customers)

# or to merge multiple DataFrames

new_df = orders.merge(customers)

# Can be extended to more than two DataFrames, just chain the commands
new_df = orders.merge(customers).merge(products).merge(salespeople)...etc
``` 

<br>

## Merge on Specific Columns I
- In the previous example, the `.merge()` function “knew” how to combine tables based on the columns that were the same between two tables.
- For instance, `products` and `orders` both had a column called `product_id`. This won’t always be true when we want to perform a merge.

```py
pd.merge(orders, customers.rename(columns={'id': 'customer_id'}))
```

- One way that we could address this problem is to use `.rename()` to rename the columns for our merges.
- In the example above, we will rename the column `id` to `customer_id`, so that `orders` and `customers` have a common column for the merge.

<br>

## Merge on Specific Columns II
- In the previous exercise, we learned how to use `.rename()` to merge two DataFrames whose columns don’t match.
- If we don’t want to do that, we have another option.
-  We could use the keywords `left_on` and `right_on` to specify which columns we want to perform the merge on.

```py
pd.merge(orders, customers, left_on='customer_id', right_on='id')
```

- If we use this syntax, we’ll end up with two columns called `id`, one from the first table and one from the second. 
-  Pandas won’t let you have two columns with the same name, so it will change them to `id_x` and `id_y`.
- he new column names `id_x` and `id_y` aren’t very helpful for us when we read the table.
- We can help make them more useful by using the keyword `suffixes`. We can provide a list of suffixes to use instead of “_x” and “_y”.

```py
pd.merge(orders, customers, left_on='customer_id', right_on='id', suffixes=['_order', '_customer'])
```

- This will change the columns `id_x` and `id_y` to `id_order` and `id_customer`, respectively.

<br>

## Outer Merge
- In the previous exercise, we saw that when we merge two DataFrames whose rows don’t match perfectly, we lose the unmatched rows.
- This type of merge (where we only include matching rows) is called an *inner merge*.
- Suppose that two companies, Company A and Company B have just merged. They each have a list of customers, but they keep slightly different data. Company A has each customer’s name and email. Company B has each customer’s name and phone number. They have some customers in common, but some are different.

```py
# Company A
>>>
         name	                  email
Sally Sparrow	sally.sparrow@gmail.com
  Peter Grant	       pgrant@yahoo.com
   Leslie May	   leslie_may@gmail.com

# Company B
>>>
       name	            phone
Peter Grant	     212-345-6789
 Leslie May	     626-987-6543
 Aaron Burr	     303-456-7891

pd.merge(company_a, company_b, how='outer')

# Output
>>>
         name	                  email	        phone
Sally Sparrow   sally.sparrow@gmail.com	          nan
Peter Grant	           pgrant@yahoo.com	 212-345-6789
Leslie May	       leslie_may@gmail.com	 626-987-6543
Aaron Burr	                        nan	 303-456-7891
```

<br>

## Left and Right Merge
- In the previous exercise, we saw that an outer merge includes all rows from both tables, even if the rows don’t have match.
- There are other types of merges that we can use to handle these cases.
    - **Left merge**: includes all rows from the first (left) table, but only rows from the second (right) table that match the first table.
    - **Right merge**: includes all rows from the second (right) table, but only rows from the first (left) table that match the second table.

```py
# Company A
>>>
         name	                  email
Sally Sparrow	sally.sparrow@gmail.com
  Peter Grant	       pgrant@yahoo.com
   Leslie May	   leslie_may@gmail.com

# Company B
>>>
       name	            phone
Peter Grant	     212-345-6789
 Leslie May	     626-987-6543
 Aaron Burr	     303-456-7891

pd.merge(company_a, company_b, how='left')

# Output
>>>
         name	                  email	        phone
Sally Sparrow   sally.sparrow@gmail.com	          nan
Peter Grant	           pgrant@yahoo.com	 212-345-6789
Leslie May	       leslie_may@gmail.com	 626-987-6543
```

```py
# Company A
>>>
         name	                  email
Sally Sparrow	sally.sparrow@gmail.com
  Peter Grant	       pgrant@yahoo.com
   Leslie May	   leslie_may@gmail.com

# Company B
>>>
       name	            phone
Peter Grant	     212-345-6789
 Leslie May	     626-987-6543
 Aaron Burr	     303-456-7891

pd.merge(company_a, company_b, how='right')

# Output
>>>
         name	                  email	        phone
Peter Grant	           pgrant@yahoo.com	 212-345-6789
Leslie May	       leslie_may@gmail.com	 626-987-6543
Aaron Burr	                        nan	 303-456-7891
```

<br>

## Concatenate DataFrames
- Sometimes, a dataset is broken into multiple tables. For instance, data is often split into multiple CSV files so that each download is smaller.
- When we need to reconstruct a single DataFrame from multiple smaller DataFrames, we can use the method `pd.concat([df1, df2, df3, ...])`. This method only works if all of the columns are the same in all of the DataFrames.

```py
# df1
>>>
          name	                    email
 Katja Obinger	      k.obinger@gmail.com
Alison Hendrix	        alisonH@yahoo.com
Cosima Niehaus	   cosi.niehaus@gmail.com
 Rachel Duncan	 rachelduncan@hotmail.com

# df2
>>>
          name	                    email
     Jean Gray	       jgray@netscape.net
 Scott Summers	       ssummers@gmail.com
   Kitty Pryde	         kitkat@gmail.com
Charles Xavier	      cxavier@hotmail.com

pd.concat([df1, df2])

# Output
>>>
          name	                    email
 Katja Obinger	      k.obinger@gmail.com
Alison Hendrix	        alisonH@yahoo.com
Cosima Niehaus	   cosi.niehaus@gmail.com
 Rachel Duncan	 rachelduncan@hotmail.com
     Jean Gray	       jgray@netscape.net
 Scott Summers	       ssummers@gmail.com
   Kitty Pryde	         kitkat@gmail.com
Charles Xavier	      cxavier@hotmail.com
```