import mysql.connector
class DBConnection:
    """
    Lớp DBConnection dùng để kết nối đến database
    """
    conn = None
    cur = None
    def __init__(self):
        self.conn = mysql.connector.connect(
            user="user",
            password="1234",
            host="vps1.mtuan.dev",
            port=3306,
            database="IOT_Cardio"
        )
        self.cur = self.conn.cursor()
    def execute(self, query):
        self.cur.execute(query)
        self.conn.commit()


    
    


    