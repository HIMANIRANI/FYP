import React, { useEffect, useState } from "react";

const AdminDashboard = () => {
  const [users, setUsers] = useState([]);
  const [feedbacks, setFeedbacks] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/admin/users")
      .then(res => res.json())
      .then(setUsers);
    fetch("http://localhost:8000/api/feedback/all")
      .then(res => res.json())
      .then(setFeedbacks);
  }, []);

  const totalUsers = users.length;
  const totalPremium = users.filter(u => u.is_premium).length;
  const totalFeedbacks = feedbacks.length;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Admin Dashboard</h1>
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="bg-white p-4 rounded shadow">Total Users: {totalUsers}</div>
        <div className="bg-white p-4 rounded shadow">Premium Users: {totalPremium}</div>
        <div className="bg-white p-4 rounded shadow">Feedbacks: {totalFeedbacks}</div>
      </div>
      <h2 className="text-xl font-semibold mb-2">Users</h2>
      <div className="overflow-x-auto mb-8">
        <table className="min-w-full border border-gray-200 divide-y divide-gray-200">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-4 py-2 text-left font-semibold">Email</th>
              <th className="px-4 py-2 text-left font-semibold">Name</th>
              <th className="px-4 py-2 text-left font-semibold">Premium</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {users.map(u => (
              <tr key={u.email} className="hover:bg-gray-50">
                <td className="px-4 py-2 whitespace-nowrap">{u.email}</td>
                <td className="px-4 py-2 whitespace-nowrap">{u.firstName} {u.lastName}</td>
                <td className="px-4 py-2 whitespace-nowrap">{u.is_premium ? "Yes" : "No"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h2 className="text-xl font-semibold mb-2">Feedback</h2>
      <table className="w-full">
        <thead>
          <tr>
            <th>Email</th><th>Message</th>
          </tr>
        </thead>
        <tbody>
          {feedbacks.map((f, i) => (
            <tr key={i}>
              <td>{f.email}</td>
              <td>{f.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AdminDashboard; 