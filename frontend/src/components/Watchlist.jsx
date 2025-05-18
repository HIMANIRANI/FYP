import React, { useEffect, useState } from "react";
import axios from "axios";

const Watchlist = () => {
  const [watchlist, setWatchlist] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch watchlist on mount
  useEffect(() => {
    const fetchWatchlist = async () => {
      const token = localStorage.getItem("access_token");
      if (!token) {
        setLoading(false);
        setWatchlist([]);
        return;
      }
      try {
        const res = await axios.get("http://localhost:8000/watchlist/get", {
          headers: { Authorization: `Bearer ${token}` },
        });
        setWatchlist(res.data);
      } catch (err) {
        if (err.response && err.response.status === 401) {
          console.error("Unauthorized: Please log in again.");
        } else {
          console.error("Failed to fetch watchlist", err);
        }
        setWatchlist([]);
      } finally {
        setLoading(false);
      }
    };
    fetchWatchlist();
  }, []);

  if (loading) return <div>Loading...</div>;
  const token = localStorage.getItem("access_token");
  if (!token) return <div>Please log in to view your watchlist.</div>;

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



