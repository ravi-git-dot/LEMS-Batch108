## Python OOPS- Inheritance

### All the class variables are public
### Car Blueprint
class Car():
    def __init__(self,windows,doors,enginetype):
        self.windows=windows
        self.doors=doors
        self.enginetype=enginetype
    def drive(self):
        print("The Person drives the car")

car = Car(4,5,"Diesel")
print(car.drive())


class audi(Car):
    def __init__(self, windows, doors, enginetype, enableai):
        super().__init__(windows, doors, enginetype)
        self.enableai = enableai

    def selfdriving(self):
        print("Audi supports Self driving")

audiQ7=audi(5,5,"diesel",True)
audiQ7.drive()
audiQ7.selfdriving()