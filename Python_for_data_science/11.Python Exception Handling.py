##Python Exception Handling

try:
    ##code block where exception can occur
    a=1
    b="s"
    c=a+b
except NameError as ex1:
    print("The user have not defined the variable")
except Exception as ex:
    print(ex)

a = b

a=1
b="s"
c=a+b

try:
    ##code block where exception can occur
    a = int(input("Enter the number 1 "))
    b = int(input("Enter the number 2 "))
    c = a / b
    d = a * b
    e = a + b

except NameError:
    print("The user have not defined the variable")
except ZeroDivisionError:
    print("Please provide number greater than 0")
except TypeError:
    print("Try to make the datatype similar")
except Exception as ex:
    print(ex)
else:
    print(c)
    print(d)
    print(e)