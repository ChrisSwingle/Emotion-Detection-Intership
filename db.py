import sqlite3
from sqlite3 import Error
from dprint import dprint
import pandas as pd
from pandas import DataFrame


class DataBase:
    def __init__(self):
        self.connection = self.sql_connection()

    def sql_connection(self, name='mydatabase.db'):

        try:
            connection = sqlite3.connect(name, check_same_thread=False)
            # print('database connected')
            return connection

        except Error:

            print(Error)

    def create_table(self):

        cursorObj = self.connection.cursor()
        try:
            # cursorObj.execute('DROP TABLE monitor')
            cursorObj.execute(
                "CREATE TABLE monitor(id INTEGER PRIMARY KEY, calm REAL, happy REAL, sad REAL, angry REAL, fearful REAL, disgust REAL, surprised REAL, positive REAL, negative REAL)")

            cursorObj.commit()
        except Error:

            print('create_table:', Error)

    def commit_sql(self):
        self.connection.commit()

    def entry(self, key, calm, happy, sad, angry, fearful, disgust, surprised, positive, negative):
        # sql = 'INSERT INTO monitor({0},{1},{2},{3},{4},{5}) VALUES(?,?,?,?,?,?)'.format(key, calm, happy, sad, angry, fearful, disgust, surprised, positive, negative)

        values = [key, calm, happy, sad, angry, fearful, disgust, surprised, positive, negative]
        # print('val',values)
        sql = ''
        if calm != None:
            sql = 'UPDATE monitor SET calm = ' + str(calm) + ' WHERE id = ' + str(key)
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
            except Error:
                self.connection.rollback()
                print('calm ', end='')
                dprint(Error)
        elif happy != None:
            sql = 'UPDATE monitor SET happy = ' + str(happy) + ' WHERE id = ' + str(key)

        elif sad != None:
            sql = 'UPDATE monitor SET sad = ' + str(sad) + ' WHERE id = ' + str(key)
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
            except Error:
                self.connection.rollback()
                print('Sad ', end='')
                dprint(Error)
        elif angry != None:
            sql = 'UPDATE monitor SET angry = ' + str(angry) + ' WHERE id = ' + str(key)
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
            except Error:
                self.connection.rollback()
                print('Angry ', end='')
                dprint(Error)
        elif fearful != None:
            sql = 'UPDATE monitor SET fearful = ' + str(fearful) + ' WHERE id = ' + str(key)
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
                print("entry submitted")
            except Error:
                self.connection.rollback()
                print('Fearful ', end='')
                dprint(Error)
        elif disgust != None:
            sql = 'UPDATE monitor SET disgust = ' + str(disgust) + ' WHERE id = ' + str(key)
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
            except Error:
                self.connection.rollback()
                print('Disgust', end='')
                dprint(Error)
        elif surprised != None:
            sql = 'UPDATE monitor SET surprised = ' + str(surprised) + ' WHERE id = ' + str(key)
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
            except Error:
                self.connection.rollback()
                print('calm', end='')
                dprint(Error)
        elif positive != None:
            sql = 'UPDATE monitor SET positive = ' + str(positive) + ' WHERE id = ' + str(key)
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
            except Error:
                self.connection.rollback()
                print('calm', end='')
                dprint(Error)
        elif negative != None:
            sql = 'UPDATE monitor SET negative = ' + str(negative) + ' WHERE id = ' + str(key)
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
            except Error:
                self.connection.rollback()
                print(Error)
        else:
            print("nothing updated")

        print(sql)

    def entry_first_phrase(self, key, calm, happy, sad, angry, fearful, disgust, surprised, positive, negative):

        cur = self.connection.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO monitor (id, calm, happy, sad, angry, fearful, disgust, surprised, positive, negative) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, calm, happy, sad, angry, fearful, disgust, surprised, positive, negative))
        self.connection.commit()

    def delete_all_tasks(self):
        """
        Delete all rows in the tasks table
        :param conn: Connection to the SQLite database
        :return:
        """
        sql = 'DELETE FROM monitor'
        cur = self.connection.cursor()
        cur.execute(sql)
        self.connection.commit()

    def get_dataframe(self):
        df = DataFrame(pd.read_sql_query('SELECT * FROM monitor', sqlite3.connect('mydatabase.db')))
        return df

    def get_transcript(self):
        df = DataFrame(pd.read_sql_query('SELECT phrase FROM monitor', sqlite3.connect('mydatabase.db')))
        return df

