import React from 'react';
import Navbar from './Navbar';

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar className="sticky top-0 z-50 w-full" />
      <main className="flex-1 w-full">
        {children}
      </main>
    </div>
  );
};

export default Layout; 