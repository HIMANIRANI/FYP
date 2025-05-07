import axios from "axios";
import React, { useEffect, useRef, useState } from "react";

const API_BASE = "http://localhost:8000"; 

export default function Watchlist({ userId }) {
  const [allStocks, setAllStocks] = useState([]);
  const [search, setSearch] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);

  // Load today's data
  useEffect(() => {
    axios.get("/data/initial/date/today.json")
      .then(res => setAllStocks(res.data.data))
      .catch(() => setAllStocks([]));
  }, []);

  // Load user's watchlist
  useEffect(() => {
    if (userId) {
      axios.get(`${API_BASE}/watchlist?user_id=${userId}`)
        .then(res => setWatchlist(res.data))
        .catch(() => setWatchlist([]));
    }
  }, [userId]);

  // Autocomplete suggestions
  useEffect(() => {
    if (search.length < 1) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }
    const filtered = allStocks.filter(stock =>
      stock.company.code.toLowerCase().includes(search.toLowerCase())
    ).slice(0, 10);
    setSuggestions(filtered);
    setShowSuggestions(filtered.length > 0);
  }, [search, allStocks]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target) &&
        inputRef.current &&
        !inputRef.current.contains(event.target)
      ) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  // Add to watchlist
  const addToWatchlist = (stock) => {
    axios.post(`${API_BASE}/watchlist/add`, {
      user_id: userId,
      scrip: stock.company.code,
      company_name: stock.company.name,
      price: stock.price,
      numTrans: stock.numTrans,
      tradedShares: stock.tradedShares,
      amount: stock.amount,
    })
      .then(res => setWatchlist([...watchlist, res.data]))
      .catch(err => {
        if (err.response && err.response.data.detail === "Scrip already in watchlist") {
          alert("Already in watchlist");
        }
      });
    setSearch("");
    setSuggestions([]);
    setShowSuggestions(false);
  };

  // Delete from watchlist
  const deleteFromWatchlist = (id) => {
    axios.delete(`${API_BASE}/watchlist/${id}`)
      .then(() => setWatchlist(watchlist.filter(item => item._id !== id)));
  };

  return (
    <div className="max-w-2xl mx-auto mt-8 p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">My Watchlist</h2>
      <div className="relative mb-6">
        <input
          ref={inputRef}
          type="text"
          placeholder="Search stock code..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          onFocus={() => setShowSuggestions(suggestions.length > 0)}
          className="w-full border border-gray-300 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 text-lg"
        />
        {showSuggestions && suggestions.length > 0 && (
          <div ref={dropdownRef} className="absolute left-0 mt-2 w-full z-20">
            {/* Arrow */}
            <div className="absolute -top-2 left-6 w-4 h-4 bg-gray-900 rotate-45 z-30"></div>
            <ul className="bg-gray-900 rounded-lg shadow-lg py-2 px-0 text-white max-h-72 overflow-y-auto">
              {suggestions.map(stock => (
                <li
                  key={stock.company.code}
                  onClick={() => addToWatchlist(stock)}
                  className="px-6 py-3 cursor-pointer hover:bg-gray-700 text-lg transition-colors"
                >
                  {stock.company.code}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <table className="min-w-full mt-4 border rounded-lg overflow-hidden shadow text-sm">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-4 py-2">Code</th>
            <th className="px-4 py-2">Company</th>
            <th className="px-4 py-2">Open</th>
            <th className="px-4 py-2">Close</th>
            <th className="px-4 py-2">Diff</th>
            <th className="px-4 py-2">Traded Shares</th>
            <th className="px-4 py-2">Amount</th>
            <th className="px-4 py-2">Delete</th>
          </tr>
        </thead>
        <tbody>
          {watchlist.map(item => (
            <tr key={item._id} className="border-t hover:bg-gray-50">
              <td className="px-4 py-2 font-semibold">{item.scrip}</td>
              <td className="px-4 py-2">{item.company_name}</td>
              <td className="px-4 py-2">{item.price.open}</td>
              <td className="px-4 py-2">{item.price.close}</td>
              <td className="px-4 py-2" style={{ color: item.price.diff >= 0 ? "green" : "red" }}>
                {item.price.diff}
              </td>
              <td className="px-4 py-2">{item.tradedShares}</td>
              <td className="px-4 py-2">{item.amount}</td>
              <td className="px-4 py-2">
                <button onClick={() => deleteFromWatchlist(item._id)} className="text-red-500 hover:underline">Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

