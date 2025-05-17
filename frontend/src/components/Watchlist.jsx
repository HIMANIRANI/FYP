import React, { useEffect, useState } from "react";
import axios from "axios";

const Watchlist = () => {
  const [watchlist, setWatchlist] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch watchlist on mount
  useEffect(() => {
    const fetchWatchlist = async () => {
      try {
        const token = localStorage.getItem("access_token");
        const res = await axios.get("http://localhost:8000/watchlist/get", {
          headers: { Authorization: `Bearer ${token}` },
        });
        setWatchlist(res.data);
      } catch (err) {
        console.error("Failed to fetch watchlist", err);
      } finally {
        setLoading(false);
      }
    };
    fetchWatchlist();
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div>
      <h2>Your Watchlist</h2>
      <ul>
        {watchlist.map((item) => (
          <li key={item.scrip.code}>
            {item.scrip.code} - {item.scrip.name} (Close: {item.scrip.price.close})
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Watchlist;


