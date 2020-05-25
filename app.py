import signal
import time
import os
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import smp
from threading import Thread
from threading import Condition
from datetime import datetime as dt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock_tracker.db'
db = SQLAlchemy(app)


def update_price(stock):
    print(f"Updating {stock.ticker} {stock.price} {stock.last_checked}")
    stock.price = smp.get_current_price_msn(stock.ticker)
    stock.last_checked = time.time()
    stock.last_check_display = datetime.utcnow()
    print(f"New price {stock.ticker} {stock.price} {stock.last_checked}")
    return stock


class TrackStock(db.Model):
    ticker = db.Column(db.String(5), primary_key=True)
    name = db.Column(db.String(30), nullable=False, default="Needs name")
    frequency = db.Column(db.Integer, default=1)
    last_checked = db.Column(db.Float, default=0.00)
    last_check_display = db.Column(db.DateTime, default=datetime.utcnow())
    price = db.Column(db.Float, default=0)

    def __repr__(self):
        return f"Added {self.ticker} {self.last_checked} {self.price}"

    # def __init__(self, ticker, frequency=1):
    #     self.ticker = ticker
    #     self.name = "need get name"
    #     self.frequency = frequency
    #     self.price = get_current_price_msn(ticker)
    #     self.last_checked = time.time()
    #     self.last_check_display = datetime.utcnow()
    #     self.last_checked = float(time.time())
    #     print(f"Added {self.ticker} {self.last_checked} {self.price}")



def seconds_until_market_open():
    time_now = dt.now()
    t_min = time_now.minute
    t_hr = time_now.hour
    if market_open['hour'] < t_hr < market_close['hour']:
        return 0
    if market_open['hour'] == t_hr and market_open['minute'] <= t_min:
        return 0
    if market_close['hour'] == t_hr and market_close['minute'] >= t_min:
        return 0
    hours = 0
    minutes = 0
    adjusted_hour = time_now.hour
    print(f"time is {time_now.hour}:{time_now.minute}")
    if t_min > market_close['minute']:
        minutes += 60 - t_min
        minutes += market_open['minute']
        adjusted_hour += 1
    if t_min < market_open['minute']:
        minutes += market_open['minute'] - time_now.minute
    if adjusted_hour > market_close['hour']:
        hours += 24 - adjusted_hour + market_open['hour']
    if adjusted_hour < market_open['hour']:
        hours += market_open['hour'] - adjusted_hour
    print(f"market closed, waiting {hours} hours and {minutes} minutes")
    return ((minutes + hours * 60) * 60)


def alarm_worker(thread_num, frequency):
    last_save = 0
    need_commit = False
    first_time = True
    calender_day = None
    while True:
        if first_time:
            # find the 00 seconds
            time.sleep(60 - dt.now().second)
            # find the minute for the frequency
            time.sleep((frequency - (dt.now().minute % frequency)) * 60)
            calender_day = dt.now().isoweekday()
            if calender_day > 5:
                time.sleep((24 - dt.now().hour + 6) * 60 * 60)
                continue
            first_time = False
        else:
            time.sleep(frequency * 60 - (time.time() - start_time))
            seconds_to_open = seconds_until_market_open()
            print(f"seconds to open {seconds_to_open}")
            if seconds_to_open > 0:
                time.sleep(seconds_to_open)
                first_time = True
                continue
        start_time = time.time()
        stocks = TrackStock.query.filter(TrackStock.frequency == frequency).all()
        print(f"Getting prices for {len(stocks)}")
        for stock in stocks:
            print(f"Checking {stock.ticker}")
        for stock in stocks:
            try:
                stock = update_price(stock)
                need_commit = True
                print(f"{stock.ticker} updated ${stock.price} by thread {thread_num} with freq {frequency}\n")
                ## todo: save price to csv
            except:
                print(f"Error: Checking {stock.last_checked} with current time {dt.now().hour}:{dt.now().minute}:{dt.now().second}")
        if need_commit:
            db.session.commit()
            need_commit = False
        print(f"Finished\n\n")


# initialize
market_open = {"hour": 7, "minute": 29}
market_close = {"hour": 14, "minute": 2}
frequencies = [1, 5, 10, 15, 30, 60]
price_thread = []
# thread_ctrl = Condition()
for i in range(len(frequencies)):
    price_thread.append(Thread(target=alarm_worker, args=(i, frequencies[i]), daemon=True).start())

# # signal.signal(signal.SIGALRM, _handle_minute_alarm)
# signal.signal(signal.alarm(), _handle_minute_alarm)
# while True:
#     if dt.now().second == 0:
#         signal.alarm(60)
#         break


@app.route('/', methods=['POST', 'GET'])
def index():
    # for stock in TrackStock.query.all():
    #     stock.price = get_current_price_msn(stock.ticker)
    if request.method == 'POST':
        ticker = request.form['ticker']
        if request.form['frequency'] is not None:
            freq = int(request.form['frequency'])
        else:
            freq = 1
        if freq < 1:
            freq = 1
        if len(ticker) > 0:
            price = smp.get_current_price_msn(ticker)
            company_info = smp.get_company_info(ticker)
            print(f"Adding {ticker} at price {price}")
            new_stock = TrackStock(ticker=ticker, frequency=freq, price=price, last_checked=time.time(), \
                                   name=company_info['name'] if 'name' in company_info else None)
            print(f"Added {new_stock.ticker} {new_stock.last_checked} {new_stock.price}")
            try:
                db.session.add(new_stock)
                db.session.commit()
                return redirect('/')
            except:
                return 'There was an issue adding your task'
    else:
        stocks = TrackStock.query.order_by(TrackStock.price).all()
        for stock in stocks:
            db.session.delete(stock)
        return render_template('index.html', stocks=stocks)


@app.route('/delete/<id>')
def delete(id):
    ticker_to_delete = TrackStock.query.get_or_404(id)

    try:
        print(f"trying to delete {id}")
        db.session.delete(ticker_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that task'


@app.route('/update/<id>', methods=['GET', 'POST'])
def update(id):
    ticker = TrackStock.query.get_or_404(id)
    print(f"ticker {ticker}")
    if request.method == 'POST':
        ticker.price = smp.get_current_price_msn(request.form['ticker'])

        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue updating your task'

    else:
        return redirect('/')
        # return render_template('update.html', stock=ticker)


if __name__ == "__main__":
    app.run(debug=True)
