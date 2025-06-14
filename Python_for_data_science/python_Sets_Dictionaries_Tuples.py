# Set
#It's similar to mathematical sets and is very useful when you need to eliminate duplicates or perform operations
# like union, intersection, etc.

# Unordered: The elements have no fixed position.
# Mutable: You can add or remove elements.
# No duplicates: Each element must be unique.
# Unindexed: You can't access elements by position (like set[0] is invalid)

my_set = {1, 2, 3, 4}
print(my_set)

my_set = set([1, 2, 2, 3])
print(my_set)

set_var = set()
print(set_var)
print(type(set_var))

set_var= {1,2,3,3,4,5}
print(set_var)

set_var = {"Avengers", "IronMan", "SuperMan", "BatMan"}
print(set_var)
print(type(set_var))

set_var.add("Hulk")
print(set_var)

# Common add an element
my_set.add(5)
print(my_set)

# Raises an error if 3 is not in the set
my_set.remove(3)
# Does nothing if 10 is not in the set
my_set.discard(10)

print(2 in my_set)

set1 = {"Avengers", "Hitman", "Ironman"}
set2 = {"Avengers", "Hitman", "Ironman", "Hulk2"}

# difference
set2.difference(set1)

# difference update in set
set2.difference_update(set1)

print(set2)

# Set Methods for Mathematical Operations

a = {1, 2, 3}
b = {3, 4, 5}

# union
print(a | b)
print(a.union(b))

# intersection
print(a & b)
print(a.intersection(b))

print(a - b)

print(a ^ b)

for item in my_set:
    print(item)

set1 = {"Avengers", "Hitman", "Ironman"}
set2 = {"Avengers", "Hitman", "Ironman", "Hulk2"}

set2.intersection(set1)

print(set2)

#   Dictionaries
#Key-value pairs (like a real dictionary: word → meaning).

#Keys must be unique and immutable (e.g., strings, numbers, tuples).

#Values can be of any data type (strings, lists, other dictionaries, etc.).

#Mutable: You can add, update, or remove items.

#Unordered in older versions of Python, but maintains insertion order

person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

print(person["name"])

print(person.get("country", "Not found"))

# Adding or updating

person["email"] = "alice@example.com"
person["age"] = 26

print(person)

# remove
person.pop("city")

del person["age"]

person.clear()

for key in person:
    print(key, person[key])

person = dict(name="Alice", age=25, city="New York")

for key in person:
    print(key, person[key])

for key, value in person.items():
    print(key, "->", value)

print(person.keys())      # dict_keys(['name', 'email'])
print(person.values())    # dict_values(['Alice', 'alice@example.com'])
print(person.items())     # dict_items([('name', 'Alice'), ('email', 'alice@example.com')])

text = "apple banana apple orange banana apple"
words = text.split()

word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1

print(word_count)

dic = {}

type(dic)

dic = {1,2,3,4,5,6,7}

type(dic)

# let create a dictorinary
my_dic = {"Car": "audi", "Car2": "BMW", "Car3": "Rollas Royals"}

type(my_dic)

print(my_dic["Car"])

for x in my_dic:
    print(x)

for x in my_dic.values():
    print(x)

for x in my_dic.items():
    print(x)

my_dic["Car4"] = "BMW 6 series"

print(my_dic)

my_dic["Car"] ="Maruthi suski"

print(my_dic)

# Nested Dictionary

Car1_model = {"Mercedes":1960}
Car2_model = {"Audi": 1970}
Car3_model = {"Ambassador": 1980}

car_type = {"Car1": Car1_model, "Car2": Car2_model, "Car3": Car3_model}

print(car_type)

print(car_type["Car1"])

print(car_type["Car1"]["Mercedes"])

# Tuples

# A tuple is a collection of ordered and immutable (unchangeable) elements.

# It can hold multiple data types.

# Tuples are defined using parentheses ().

# create an empty tuples

my_tuple = tuple()

type(my_tuple)

my_tuple = ()

my_tuple = ("Krish", "Ankur", "John")

print(my_tuple[0])

print(type(my_tuple))
print(my_tuple)

type(my_tuple)

# inbulit Function
my_tuple.count("Krish")

# index not possible
my_tuple[0]= "Ram"

my_tuple.count("Ankur")