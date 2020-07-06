import smtplib
from email.utils import formatdate
from email.mime.text import MIMEText


def send_email(msg, passwd, from_addr, to_addr=None):
    """You should allow "Less secure app access" in Google acount.
    https://myaccount.google.com/u/1/lesssecureapps?pageId=none
    """
    to_addr = to_addr if to_addr else from_addr
    
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.ehlo()
    smtpobj.starttls()
    smtpobj.ehlo()
    smtpobj.login(from_addr, passwd)

    msg = MIMEText(msg)
    msg['Subject'] = "Message from GPU server"
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Date'] = formatdate()

    smtpobj.sendmail(from_addr, to_addr, msg.as_string())
    smtpobj.close()