"use client";
import React, { useState } from 'react';
import { Stethoscope, Menu, X, Home, Info, Phone } from 'lucide-react';

interface NavBarProps {
    currentPage: string;
    setCurrentPage: (page: string) => void;
}

const NavBar = ({ currentPage, setCurrentPage }: NavBarProps) => {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    return (
        <nav className="bg-white shadow-md sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-16">
                    <div className="flex items-center space-x-2">
                        <div className="bg-gradient-to-br from-teal-500 to-blue-600 p-2 rounded-lg">
                            <Stethoscope className="text-white" size={24} />
                        </div>
                        <span className="text-xl font-bold bg-gradient-to-r from-teal-600 to-blue-600 bg-clip-text text-transparent">
                            GVS
                            <sup className='bg-gradient-to-r from-teal-600 to-blue-600 bg-clip-text text-transparent'>2</sup>
                        </span>
                    </div>

                    <div className="hidden md:flex space-x-8">
                        <button
                            onClick={() => setCurrentPage('home')}
                            className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-all ${currentPage === 'home'
                                    ? 'text-teal-600 bg-teal-50'
                                    : 'text-gray-600 hover:text-teal-600 hover:bg-gray-50'
                                }`}
                        >
                            <Home size={18} />
                            <span>Home</span>
                        </button>
                        <button
                            onClick={() => setCurrentPage('about')}
                            className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-all ${currentPage === 'about'
                                    ? 'text-teal-600 bg-teal-50'
                                    : 'text-gray-600 hover:text-teal-600 hover:bg-gray-50'
                                }`}
                        >
                            <Info size={18} />
                            <span>About</span>
                        </button>
                        <button
                            onClick={() => setCurrentPage('contact')}
                            className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-all ${currentPage === 'contact'
                                    ? 'text-teal-600 bg-teal-50'
                                    : 'text-gray-600 hover:text-teal-600 hover:bg-gray-50'
                                }`}
                        >
                            <Phone size={18} />
                            <span>Contact</span>
                        </button>
                    </div>

                    <button
                        className="md:hidden"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    >
                        {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>
                </div>

                {mobileMenuOpen && (
                    <div className="md:hidden pb-4 space-y-2">
                        <button
                            onClick={() => {
                                setCurrentPage('home');
                                setMobileMenuOpen(false);
                            }}
                            className="flex items-center space-x-2 w-full px-3 py-2 rounded-lg text-gray-600 hover:bg-teal-50 hover:text-teal-600"
                        >
                            <Home size={18} />
                            <span>Home</span>
                        </button>
                        <button
                            onClick={() => {
                                setCurrentPage('about');
                                setMobileMenuOpen(false);
                            }}
                            className="flex items-center space-x-2 w-full px-3 py-2 rounded-lg text-gray-600 hover:bg-teal-50 hover:text-teal-600"
                        >
                            <Info size={18} />
                            <span>About</span>
                        </button>
                        <button
                            onClick={() => {
                                setCurrentPage('contact');
                                setMobileMenuOpen(false);
                            }}
                            className="flex items-center space-x-2 w-full px-3 py-2 rounded-lg text-gray-600 hover:bg-teal-50 hover:text-teal-600"
                        >
                            <Phone size={18} />
                            <span>Contact</span>
                        </button>
                    </div>
                )}
            </div>
        </nav>
    );
}

export default NavBar;
