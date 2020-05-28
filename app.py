import signal
import time
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import smp
from threading import Thread
from threading import Condition
from datetime import datetime as dt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///track_list.db'
db = SQLAlchemy(app)


def update_price(stock):
    print(f"Updating {stock.ticker} {stock.price} {stock.last_checked}")
    time_checked = dt.now()
    stock.price = smp.get_current_price_msn(stock.ticker)
    stock.last_checked = f"{time_checked.hour}:{time_checked.minute} {time_checked.day} {time_checked.month} {time_checked.year}"
    stock.last_check_display = datetime.utcnow()
    print(f"New price {stock.ticker} {stock.price} {stock.last_checked}")
    return stock


class TrackStock(db.Model):
    ticker = db.Column(db.String(5), primary_key=True)
    name = db.Column(db.String(30), nullable=False, default="Needs name")
    frequency = db.Column(db.Integer, default=1)
    last_checked = db.Column(db.String(40), default="unknown")
    price = db.Column(db.Float, default=0)

    def __repr__(self):
        return f"Added {self.ticker} {self.last_checked} {self.price}"


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
    return (minutes + hours * 60) * 60


def current_time_str():
    if dt.now().hour < 10:
        hrs = f"0{dt.now().hour}"
    else:
        hrs = dt.now().hour
    if dt.now().minute < 10:
        mins = f"0{dt.now().minute}"
    else:
        mins = dt.now().minute
    return f"{hrs}:{mins}"


def save_prices(ticker, frequency, prices, date, month=dt.now().month, year=dt.now().year,
                location=os.path.join("data", "prices")):
    try:
        if not os.path.isdir(location):
            os.mkdir(location)
        filename = f"{ticker}_{month}_{year}_freq_{frequency}.csv"
        file_path = os.path.join(location, filename)
        if os.path.isfile(file_path):
            prices.loc[[date]].to_csv(file_path, mode='a', header='False')
        else:
            prices.to_csv(file_path)
    except Exception as e:
        print(f"Error saving prices to file for {file_path}: {e}")


def alarm_worker(thread_num, frequency, ctrl):
    last_save = 0
    need_commit = False
    first_time = True
    calender_day = None
    day_prices = {}
    current_date = ""
    blank_template = {}
    time_range = pd.timedelta_range(start=f"{market_open['hour']}:{market_open['minute']}:00",
                                    end=f"{market_close['hour']}:{market_close['minute']}:00",
                                    freq=f'{frequency}MIN')
    col_names = []
    col_names.append('date')
    for i in range(len(time_range)):
        col_names.append(str(time_range[i])[-8:-3])
    blank_df = pd.DataFrame(columns=col_names)
    blank_df.set_index('date', inplace=True)

    while True:
        if first_time:
            # find the 00 seconds
            time.sleep(60 - dt.now().second)
            # find the minute for the frequency
            time.sleep((frequency - (dt.now().minute % frequency)) * 60)
            calender_day = dt.now().isoweekday()
            if calender_day > 5:
                time.sleep((24 - dt.now().hour + 4) * 60 * 60)
                continue
            first_time = False
            current_date = f"{dt.now().day}/{dt.now().month}/{dt.now().year}"
        else:
            time.sleep(frequency * 60 - (time.time() - start_time))
            seconds_to_open = seconds_until_market_open()
            print(f"seconds to open {seconds_to_open}")
            if seconds_to_open > 0:
                for stock_ticker in day_prices.keys():
                    save_prices(stock_ticker, frequency, day_prices[stock_ticker], current_date)
                seconds_to_open = seconds_until_market_open()
                time.sleep(seconds_to_open)
                first_time = True
                continue
        start_time = time.time()
        time_key = current_time_str()
        stocks = TrackStock.query.filter(TrackStock.frequency == frequency).all()
        if len(stocks > 0):
            print(f"Getting prices for {len(stocks)}")
            for stock in stocks:
                print(f"Checking {stock.ticker}")
            for stock in stocks:
                try:
                    stock = update_price(stock)
                    need_commit = True
                    print(f"{stock.ticker} updated ${stock.price} by thread {thread_num} with freq {frequency}\n")
                    if stock.ticker not in day_prices:
                        day_prices[stock.ticker] = blank_df
                    day_prices[stock.ticker].loc[current_date, time_key] = stock.price
                except:
                    print(f"Error: Checking {stock.last_checked} with current time {time_key}")
            if need_commit:
                db.session.commit()
                need_commit = False
            print(f"Finished\n")
        else:
            print(f"No stocks at freq {frequency}, waiting")
            ctrl.wait()
            print(f"New stocks, checking")


# initialize
market_open = {"hour": 7, "minute": 30}
market_close = {"hour": 14, "minute": 00}
frequencies = [1, 5, 10, 15, 30, 60]
price_thread = []
thread_ctrl = Condition()
for i in range(len(frequencies)):
    price_thread.append(Thread(target=alarm_worker, args=(i, frequencies[i], thread_ctrl), daemon=True).start())

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
    stock = TrackStock.query.get_or_404(id)
    print(f"ticker is {stock.ticker}")
    if request.method == 'POST':
        stock.price = smp.get_current_price_msn(request.form['ticker'])

        try:
            db.session.commit()
            thread_ctrl.notify_all()
            return redirect('/')
        except:
            return 'There was an issue updating your task'

    else:
        # return redirect('/')
        stock_info = smp.load_financials(stock.ticker)
        print(f"info is {str(stock_info)[:100]}")
        return render_template('update.html', stock=stock, stock_info=stock_info)


if __name__ == "__main__":
    app.run(debug=True)
