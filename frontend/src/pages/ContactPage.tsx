"use client";
import React from 'react';
import { Phone, User, Info } from 'lucide-react';

const ContactPage = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-teal-50 to-cyan-50 py-12">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-8">Contact Us</h1>

            <div className="bg-white rounded-2xl shadow-xl p-8">
                <div className="grid md:grid-cols-2 gap-8 mb-8">
                    <div>
                        <h3 className="text-xl font-semibold text-gray-800 mb-4">Get in Touch</h3>
                        <div className="space-y-4">
                            <div className="flex items-start space-x-3">
                                <Phone className="text-teal-600 mt-1" size={20} />
                                <div>
                                    <p className="font-medium text-gray-800">Phone</p>
                                    <p className="text-gray-600">+1 (555) 123-4567</p>
                                </div>
                            </div>
                            <div className="flex items-start space-x-3">
                                <User className="text-teal-600 mt-1" size={20} />
                                <div>
                                    <p className="font-medium text-gray-800">Email</p>
                                    <p className="text-gray-600">support@interactivediagnostics.ai</p>
                                </div>
                            </div>
                            <div className="flex items-start space-x-3">
                                <Info className="text-teal-600 mt-1" size={20} />
                                <div>
                                    <p className="font-medium text-gray-800">Address</p>
                                    <p className="text-gray-600">123 Medical Plaza<br />Healthcare District<br />San Francisco, CA 94102</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h3 className="text-xl font-semibold text-gray-800 mb-4">Business Hours</h3>
                        <div className="space-y-2 text-gray-700">
                            <p>Monday - Friday: 9:00 AM - 6:00 PM</p>
                            <p>Saturday: 10:00 AM - 4:00 PM</p>
                            <p>Sunday: Closed</p>
                        </div>

                        <div className="mt-6">
                            <h3 className="text-xl font-semibold text-gray-800 mb-4">Technical Support</h3>
                            <p className="text-gray-600 text-sm">
                                For API integration support or technical issues, please email our technical team at
                                <a href="mailto:tech@interactivediagnostics.ai" className="text-teal-600 hover:underline ml-1">
                                    tech@interactivediagnostics.ai
                                </a>
                            </p>
                        </div>
                    </div>
                </div>

                <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-4">Send us a Message</h3>
                    <div className="space-y-4">
                        <div className="grid md:grid-cols-2 gap-4">
                            <input
                                type="text"
                                placeholder="Your Name"
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none"
                            />
                            <input
                                type="email"
                                placeholder="Your Email"
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none"
                            />
                        </div>
                        <input
                            type="text"
                            placeholder="Subject"
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none"
                        />
                        <textarea
                            placeholder="Your Message"
                            rows={6}
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none resize-none"
                        />
                        <button
                            onClick={() => alert('Message sent! We will get back to you soon.')}
                            className="w-full bg-gradient-to-r from-teal-600 to-blue-600 text-white py-3 rounded-lg font-semibold hover:from-teal-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl"
                        >
                            Send Message
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
);

export default ContactPage;