import time
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from data_collector import smp
from threading import Thread

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock_tracker.db'
db = SQLAlchemy(app)


def worker_function(thread_num):
    last_save = 0
    need_commit = False
    while True:
        for stock in TrackStock.query.all():
            try:
                if type(stock.last_checked) is not float or stock.last_checked == 0 or (
                        time.time() - stock.last_checked > (stock.frequency * 60)):
                    stock = update_price(stock)
                    need_commit = True
                    print(f"{stock.ticker} updated ${stock.price}")
            except:
                print(f"Error: Checking {stock.last_checked} with current time {time.time()}")
        if need_commit:
            db.session.commit()
            need_commit = False


def update_price(stock):
    print(f"Updating {stock.ticker} {stock.price} {stock.last_checked}")
    stock.price = smp.get_current_price_msn(stock.ticker)
    stock.last_checked = time.time()
    stock.last_check_display = datetime.utcnow()
    print(f"New price {stock.ticker} {stock.price} {stock.last_checked}")
    return stock


#
# def get_price(ticker):
#     # stock.price = get_current_price_msn(stock.ticker)
#     # stock.last_checked = time.time()
#     # stock.last_check_display = datetime.utcnow()
#     # print(f"{stock.ticker} updated ${stock.price}")
#     return get_current_price_msn(ticker), float(time.time()), datetime.utcnow()


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


# initialize
price_thread = Thread(target=worker_function, args=(1,))
price_thread.daemon = True
price_thread.start()


@app.route('/', methods=['POST', 'GET'])
def index():
    # for stock in TrackStock.query.all():
    #     stock.price = get_current_price_msn(stock.ticker)
    if request.method == 'POST':
        ticker = request.form['ticker']
        freq = int(request.form['frequency'])
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
