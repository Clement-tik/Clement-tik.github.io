<link rel="stylesheet" href="/style.css">
# CryptoForecast: A Time-Series Forecasting Prototype for Cryptocurrency Prices

* [Our Members](#our-members)
* [1. INTRODUCTION](#1-introduction)
* [2. Datasets](#2-datasets)
* [3. Methodology](#3-methodology)
* [4. Evaluation & Analysis](#4-evaluation-analysis)
* [5. Related Work](#5-related-work)
## Our Members :

* Clement Tikhomiroff ,dpt of computer sience ,Hanyang University ,clement.tikhomiroff@gmail.com
* Victor Venail ,dpt of computer science ,Hanyang University ,victor.venail@edu.ece.fr
* Elodie huang ,dpt of computer science ,Hanyang University ,elodie.huang@edu.ece.fr
* Cindy HU ,dpt of computer science ,Hanyang University ,cindy.hu@edu.ece.fr
* our blog address https://clement-tik.github.io/
## 1 INTRODUCTION  :

Predicting the evolution of financial markets is often compared to “predicting the unpredictable” a challenge that reaches its peak with cryptocurrencies. Indeed, we are talking about high volatile market which can either make you rich or makes you lose everything.

Indeed, this year, the cryptocurrency ecosystem has been shaken by exceptional volatility, far exceeding that of traditional markets. From January 1 to October 29, 2024, the Bitcoin volatility index saw extreme peaks, much higher than those of 2023. Furthermore, the threat of a "bear market" appeared with bewildering speed.
* A bear market is a situation when the market goes down by 20% or more.

In fact, a cryptocurrency's price depends on a combination of factors to start and evolve: a supply (often limited), demand, and an underlying technology. But not all price movements are created equal.

### The "Ingredients" of a Market Movement :

* A Catalyst: This is anything that can generate sudden interest. Common catalysts are an announcement from the Federal Reserve (FED) or an ETF approval.
* Market Players: This is the available liquidity and general sentiment. Retail investors, investment funds, and "whales" (large wallets) are the typical drivers. The amount of "FOMO" (Fear of Missing Out) or "FUD" (Fear, Uncertainty, and Doubt) greatly affects how quickly a price evolves.
* Asset Nature: Certain assets, like "memecoins," are propelled by social media mechanisms and can react more quickly and intensely than established assets like Bitcoin.

### The Spread of the Movement:

Once a movement starts, it can spread rapidly depending on market sentiment and the structure of the order book.
* Social Media Sentiment: This is a key factor. The volume of mentions on X (Twitter) or Reddit is a critical element. Studies have shown that a rise in positive sentiment is linked to an increase in prices. Very high sentiment can suck up liquidity and cause prices to rise.
* Algorithmic Trading ("Bots"): The rapid interventions of high-frequency trading bots bring a new influx of orders (buy/sell) to the movement. They can also propagate cascading effects, triggering liquidations in the derivatives markets.
* Structure (Order Books): Prices tend to move faster uphill as they break through "sell walls" (resistance) than downhill. This acceleration is due to buying pressure forcing stop-loss orders located further up to be triggered, allowing them to be hit more easily.

### The Consequences:

These extreme market movements represent a major problem with diverse consequences:
* Economic Issues: Volatility can cause immense losses for unwary investors. In 2022, the collapse of several platforms destroyed billions of dollars in capital, ruining thousands of investors. These collapses disrupt economic activity by destroying confidence and freezing capital.
* Human Issues: Volatility causes immense human stress. Furthermore, the unregulated nature of the market fosters scams ("rug pulls") that target individuals. Exposure to this volatility is estimated to cause mental health problems for thousands of investors.

Faced with this continuous increase in volatility, questions arise: When will it stop? Can this risk at least be managed? And if so, what can we do? How can we better protect investors and minimize losses?

Some of these losses could have been avoided, but the vast majority were inevitable due to the nature of the market. That's why the best way to reduce damage and protect people is anticipation.

To map risk zones based on market conditions. To allow investors to prepare BEFORE a violent move. To save billions of euros in savings. To minimize the destruction of capital. To allow for better risk management by professionals and individuals who are better prepared. To collect scientific data to better understand the factors influencing the onset and intensity of these movements.

That is what our project is about. To develop a program using artificial intelligence to predict the direction and intensity of a market movement, to preserve capital and reduce the frequency of losses using the collected data.

## 2 Datasets :

To do so, we need a complete and reliable dataset to train our model. So, we decided to choose Cryptocurrency Historical Prices, a dataset which contains about 23 of the most popular cryptocurrency such as BTC , ETH or SOL.

   ![an example of our dataset](images/BDD_AI_BTC.jpg)
   
We got here a screenshot of our project's dataset. It's a CSV file containing historical time-series data for Bitcoin (BTC).
This is the raw data our AI model will use for training. Each row represents a single day and includes the key features for our analysis:
* SNo : serial number
* Name : the name of the coin 
* Symbol : Symbol of coin
* Date : Date
* High : High value on the date
* Low : Low value on the date
* Open : Open value on the date
* Close : Close value on the date
* Volume : Volume of transactions in USD
* MarketCAP : the marketCAP of the coin which is : Current Price × Circulating Supply

Then we got the exact same type of files for the others coins.
## 3 Methodology :

## 4 Evaluation & Analysis :

## 5 Related Work 

 

