import sqlite3
from sqlite3 import Error
from dprint import dprint
import pandas as pd
from pandas import DataFrame


class TranscriptDataBase:

    def __init__(self):
        self.connection = self.sql_connection()

    def sql_connection(self, name='transcript_database.db'):
        try:
            connection = sqlite3.connect(name,check_same_thread=False)
            #print('database connected')
            return connection
        except Error:
            print(Error)

    def create_table(self):

        cursorObj = self.connection.cursor()
        try:
            # cursorObj.execute('DROP TABLE monitor')
            cursorObj.execute("CREATE TABLE transcript (id INTEGER PRIMARY KEY, speaker INTEGER, word TEXT)")
            print("creating transcript table")
            #cursorObj.commit()

        except Error:
            print('creating transcript table: ', Error)

    def commit_sql(self):
        self.connection.commit()

    def entry(self, speaker, word):

        sql = ''
        if speaker != None:
            sql = "INSERT INTO transcript (speaker,word) VALUES('" + str(speaker)+ "'" + "," + "'" + str(word) + "')"
            cur = self.connection.cursor()
            try:
                cur.execute(sql)
                self.connection.commit()
            except Error:
                self.connection.rollback()
                print('speaker ', end='')
                dprint(Error)
        # elif word != None:
        #     sql = 'UPDATE monitor SET word = '+str(word)+' WHERE id = ' + str(key)
        #     cur = self.connection.cursor()
        #     try:
        #         cur.execute(sql)
        #         self.connection.commit()
        #     except Error:
        #         self.connection.rollback()
        #         print('word ', end='')
        #         dprint(Error)
        else:
            print('nothing updated')

        print(sql)

    def entry_first_phrase(self,key,speaker,word):
        cur = self.connection.cursor()
        cur.execute("INSERT OR REPLACE INTO transcript (id,speaker,word) VALUES(?,?,?)",(key,speaker,word))
        self.connection.commit()

    def delete_all_tasks(self):
        sql = 'DELETE FROM transcript'
        cur = self.connection.cursor()
        cur.execute(sql)
        self.connection.commit()

    def get_dataframe(self):
        df = DataFrame(pd.read_sql_query('SELECT * FROM transcript', sqlite3.connect('transcript_database.db')))
        return df

    def get_length(self):
        sql = 'SELECT * FROM transcript'
        cur = self.connection.cursor()
        return cur.execute(sql).rowcount
