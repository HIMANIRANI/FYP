import React, { useEffect, useState } from 'react';
import './StockTicker.css';

const StockTicker = () => {
  const [stockData, setStockData] = useState([]);

  useEffect(() => {
    const companies = ['NABIL', 'NIMB', 'SCB', 'HBL', 'SBI', 'EBL', 'NICA', 'MBL', 'LSL', 'KBL', 'SBL', 'SHL'];
    
    // Fetch data for each company
    const fetchStockData = async () => {
      try {
        const data = await Promise.all(
          companies.map(async (company) => {
            const response = await fetch(`/api/data/cleaned_companydata/${company}.json`);
            const json = await response.json();
            const latestData = json[json.length - 1]; // Get the most recent data
            return {
              code: company,
              volume: latestData.tradedShares,
              closePrice: latestData.close.toFixed(2),
              priceChange: ((latestData.close - latestData.prevClose) / latestData.prevClose * 100).toFixed(2)
            };
          })
        );
        setStockData(data);
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };

    fetchStockData();
    const interval = setInterval(fetchStockData, 60000); // Refresh every minute

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="ticker-container">
      <div className="ticker">
        {stockData.map((stock, index) => (
          <div key={`${stock.code}-${index}`} className="ticker-item">
            <span className="company-code">{stock.code}</span>
            <span className="volume">Vol: {stock.volume.toLocaleString()}</span>
            <span className="price">₹{stock.closePrice}</span>
            <span className={`change ${parseFloat(stock.priceChange) >= 0 ? 'positive' : 'negative'}`}>
              {stock.priceChange}%
            </span>
          </div>
        ))}
        {/* Duplicate items for seamless loop */}
        {stockData.map((stock, index) => (
          <div key={`${stock.code}-${index}-duplicate`} className="ticker-item">
            <span className="company-code">{stock.code}</span>
            <span className="volume">Vol: {stock.volume.toLocaleString()}</span>
            <span className="price">₹{stock.closePrice}</span>
            <span className={`change ${parseFloat(stock.priceChange) >= 0 ? 'positive' : 'negative'}`}>
              {stock.priceChange}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StockTicker; 
