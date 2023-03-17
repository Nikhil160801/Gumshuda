import pandas as pd
import sqlite3
con=sqlite3.connect('CriminalDetails.db')
cur=con.cursor()
cur.execute("Select * from people;")
result=cur.fetchall()
print(result)