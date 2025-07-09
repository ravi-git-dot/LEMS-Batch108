## Python OOPS-Magic Methods In Classes

### All the class variables are public
### Car Blueprint
class Car():
    def __new__(self, windows, doors, enginetype):
        print("The object has started getting initialized")

    def __init__(self, windows, doors, enginetype):
        self.windows = windows
        self.doors = doors
        self.enginetype = enginetype

    def __str__(self):
        return "The object has been initialized"

    def __sizeof__(self):
        return "This displays size of the object"

    def drive(self):
        print("The Person drives the car")

c=Car(4,5,"Diesel")
print(c)
print(c.__sizeof__())
dir(c)