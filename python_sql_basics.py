import mysql.connector

#"pip install mysql-connect-python"

try:
    connection = mysql.connector.connect(host="127.0.0.1",
                                              port=3306,
                                              user="root",
                                              password="Ravi@762000",
                                              db="uptor_108")
    my_cursor = connection.cursor()
    my_cursor.execute("show databases")
    for db in my_cursor:
        print(db)
except Exception as ex:
    print(ex)

print(connection)
print("Mysql executed successfully")
print("Git Success")