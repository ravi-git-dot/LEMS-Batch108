'''
Python Assert
Python provides the assert statement to check if a given logical expression is true or false.
Program execution proceeds only if the expression is true and raises the AssertionError when it is false.
The following code shows the usage of the assert statement.
'''

try:
    num = int(input("Enter an even number: "))
    assert num % 2 == 0  # Raises AssertionError if number is not even
    print("The number is even.")
except AssertionError:
    print("Please enter an even number.")
except ValueError:
    print("Invalid input! Please enter a valid integer.")
