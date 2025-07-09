## Python OOPS- Public, Protected And Private

### All the class variables are public
class Car():
    def __init__(self,windows,doors,enginetype):
        self.windows=windows
        self.doors=doors
        self.enginetype=enginetype

audi=Car(4,5,"Diesel")
print(audi)
print(audi.windows)

### All the class variables are protected
class Car():
    def __init__(self,windows,doors,enginetype):
        self._windows=windows
        self._doors=doors
        self._enginetype=enginetype

class Truck(Car):
    def __init__(self,windows,doors,enginetype,horsepower):
        super().__init__(windows,doors,enginetype)
        self.horsepowwer=horsepower
truck=Truck(4,4,"Diesel",4000)
dir(truck)
print(truck.horsepowwer)

truck._doors=5
print(truck._doors)


### private
class Car():
    def __init__(self,windows,doors,enginetype):
        self.__windows=windows
        self.__doors=doors
        self.__enginetype=enginetype

audi=Car(4,4,"Diesel")
print(audi)
audi._Car__doors=5
print(audi._Car__doors)
dir(audi)