"use client";
import React from 'react';
import { FileText, Download, Printer } from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

interface ReportGeneratorProps {
    topDiseases: [string, number][];
    history: Array<{
        question: string;
        answer: string;
        symptom?: string;
    }>;
    parsedSymptoms: Record<string, number>;
    finishReason: string | null;
}

const ReportGenerator = ({ topDiseases, history, parsedSymptoms, finishReason }: ReportGeneratorProps) => {
    
    const formatDiseaseName = (disease: string): string => {
        return disease.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    };

    const formatSymptomName = (symptom: string): string => {
        return symptom.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    };

    const generatePDF = () => {
        const doc = new jsPDF();
        const pageWidth = doc.internal.pageSize.getWidth();
        const pageHeight = doc.internal.pageSize.getHeight();
        
        // Colors
        const primaryColor: [number, number, number] = [20, 184, 166]; // Teal
        const secondaryColor: [number, number, number] = [37, 99, 235]; // Blue
        const darkGray: [number, number, number] = [55, 65, 81];
        const lightGray: [number, number, number] = [243, 244, 246];
        
        // Header Background
        doc.setFillColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        doc.rect(0, 0, pageWidth, 45, 'F');
        
        // Logo/Title Section
        doc.setFontSize(28);
        doc.setTextColor(255, 255, 255);
        doc.setFont('helvetica', 'bold');
        doc.text('GVS²', 20, 25);
        
        doc.setFontSize(12);
        doc.setFont('helvetica', 'normal');
        doc.text('Medical Diagnostic Report', 20, 35);
        
        // Report Date & Time
        const now = new Date();
        const dateStr = now.toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
        const timeStr = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        doc.setFontSize(9);
        doc.setTextColor(255, 255, 255);
        doc.text(`Date: ${dateStr}`, pageWidth - 60, 25);
        doc.text(`Time: ${timeStr}`, pageWidth - 60, 32);
        
        let yPos = 55;
        
        // Patient Information Section
        doc.setFillColor(lightGray[0], lightGray[1], lightGray[2]);
        doc.roundedRect(15, yPos, pageWidth - 30, 25, 3, 3, 'F');
        
        doc.setFontSize(11);
        doc.setTextColor(darkGray[0], darkGray[1], darkGray[2]);
        doc.setFont('helvetica', 'bold');
        doc.text('PATIENT INFORMATION', 20, yPos + 8);
        
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9);
        doc.text(`Report ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}`, 20, yPos + 15);
        doc.text(`Session Type: Interactive Diagnostic Analysis`, 20, yPos + 21);
        
        yPos += 35;
        
        // Diagnostic Results Section
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        doc.text('DIAGNOSTIC RESULTS', 20, yPos);
        
        yPos += 10;
        
        // Top Diseases Table
        const diseaseData = topDiseases.map((disease, index) => [
            index + 1,
            formatDiseaseName(disease[0]),
            `${(disease[1] * 100).toFixed(2)}%`,
            disease[1] > 0.7 ? 'High' : disease[1] > 0.4 ? 'Medium' : 'Low'
        ]);
        
        autoTable(doc, {
            startY: yPos,
            head: [['Rank', 'Condition', 'Probability', 'Confidence']],
            body: diseaseData,
            theme: 'grid',
            headStyles: {
                fillColor: [primaryColor[0], primaryColor[1], primaryColor[2]],
                textColor: [255, 255, 255],
                fontStyle: 'bold',
                fontSize: 10
            },
            styles: {
                fontSize: 9,
                cellPadding: 5
            },
            columnStyles: {
                0: { cellWidth: 20, halign: 'center' },
                1: { cellWidth: 80 },
                2: { cellWidth: 35, halign: 'center' },
                3: { cellWidth: 35, halign: 'center' }
            },
            alternateRowStyles: {
                fillColor: [249, 250, 251]
            }
        });
        
        yPos = (doc as any).lastAutoTable.finalY + 15;
        
        // Reported Symptoms Section
        if (Object.keys(parsedSymptoms).length > 0) {
            doc.setFontSize(14);
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
            doc.text('REPORTED SYMPTOMS', 20, yPos);
            
            yPos += 8;
            
            const positiveSymptoms = Object.entries(parsedSymptoms)
                .filter(([_, value]) => value === 1)
                .map(([symptom, _]) => formatSymptomName(symptom));
            
            doc.setFontSize(9);
            doc.setFont('helvetica', 'normal');
            doc.setTextColor(darkGray[0], darkGray[1], darkGray[2]);
            
            const symptomsPerRow = 2;
            const colWidth = (pageWidth - 40) / symptomsPerRow;
            
            positiveSymptoms.forEach((symptom, index) => {
                const col = index % symptomsPerRow;
                const row = Math.floor(index / symptomsPerRow);
                const xPos = 20 + (col * colWidth);
                const currentY = yPos + (row * 7);
                
                // Bullet point
                doc.setFillColor(primaryColor[0], primaryColor[1], primaryColor[2]);
                doc.circle(xPos + 2, currentY - 1.5, 1, 'F');
                doc.text(symptom, xPos + 6, currentY);
            });
            
            yPos += Math.ceil(positiveSymptoms.length / symptomsPerRow) * 7 + 10;
        }
        
        // Check if we need a new page
        if (yPos > pageHeight - 80) {
            doc.addPage();
            yPos = 20;
        }
        
        // Diagnostic Session Summary
        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        doc.text('SESSION SUMMARY', 20, yPos);
        
        yPos += 10;
        
        const sessionData = history.slice(-10).map((item, index) => [
            index + 1,
            item.question,
            item.answer
        ]);
        
        autoTable(doc, {
            startY: yPos,
            head: [['#', 'Question', 'Response']],
            body: sessionData,
            theme: 'striped',
            headStyles: {
                fillColor: [primaryColor[0], primaryColor[1], primaryColor[2]],
                textColor: [255, 255, 255],
                fontStyle: 'bold',
                fontSize: 9
            },
            styles: {
                fontSize: 8,
                cellPadding: 4
            },
            columnStyles: {
                0: { cellWidth: 15, halign: 'center' },
                1: { cellWidth: 100 },
                2: { cellWidth: 55 }
            }
        });
        
        yPos = (doc as any).lastAutoTable.finalY + 15;
        
        // Conclusion Section
        if (finishReason) {
            if (yPos > pageHeight - 60) {
                doc.addPage();
                yPos = 20;
            }
            
            doc.setFillColor(lightGray[0], lightGray[1], lightGray[2]);
            doc.roundedRect(15, yPos, pageWidth - 30, 30, 3, 3, 'F');
            
            doc.setFontSize(11);
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(darkGray[0], darkGray[1], darkGray[2]);
            doc.text('CONCLUSION', 20, yPos + 8);
            
            doc.setFont('helvetica', 'normal');
            doc.setFontSize(9);
            const splitReason = doc.splitTextToSize(finishReason, pageWidth - 50);
            doc.text(splitReason, 20, yPos + 16);
            
            yPos += 35;
        }
        
        // Footer - Stamp and Disclaimer
        const footerY = pageHeight - 40;
        
        // Stamp/Seal
        doc.setDrawColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        doc.setLineWidth(2);
        doc.circle(pageWidth - 40, footerY + 10, 15, 'S');
        
        doc.setFontSize(8);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        doc.text('GVS²', pageWidth - 45, footerY + 8, { align: 'center' });
        doc.text('VERIFIED', pageWidth - 45, footerY + 13, { align: 'center' });
        
        const stampDate = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        doc.setFontSize(6);
        doc.text(stampDate, pageWidth - 45, footerY + 17, { align: 'center' });
        
        // Disclaimer
        doc.setFontSize(7);
        doc.setFont('helvetica', 'italic');
        doc.setTextColor(100, 100, 100);
        const disclaimer = 'DISCLAIMER: This report is generated by an AI-powered diagnostic system and should not be considered as a substitute for professional medical advice. Please consult with a qualified healthcare provider for proper diagnosis and treatment.';
        const splitDisclaimer = doc.splitTextToSize(disclaimer, pageWidth - 90);
        doc.text(splitDisclaimer, 20, footerY + 5);
        
        // Footer line
        doc.setDrawColor(lightGray[0], lightGray[1], lightGray[2]);
        doc.setLineWidth(0.5);
        doc.line(20, footerY - 5, pageWidth - 20, footerY - 5);
        
        // Page number
        doc.setFontSize(8);
        doc.setFont('helvetica', 'normal');
        doc.text(`Page ${doc.getCurrentPageInfo().pageNumber}`, pageWidth / 2, pageHeight - 10, { align: 'center' });
        
        // Save the PDF
        const fileName = `GVS2_Report_${now.getFullYear()}${(now.getMonth()+1).toString().padStart(2,'0')}${now.getDate().toString().padStart(2,'0')}_${now.getHours()}${now.getMinutes()}.pdf`;
        doc.save(fileName);
    };

    const printReport = () => {
        generatePDF();
        // Note: For actual printing, you'd need to open the PDF in a new window
        setTimeout(() => {
            alert('PDF generated! You can now print it using your PDF viewer.');
        }, 500);
    };

    return (
        <div className="bg-gradient-to-br from-teal-50 to-blue-50 rounded-2xl shadow-xl p-6 border-2 border-teal-200">
            <div className="flex items-center mb-4">
                <div className="bg-gradient-to-br from-teal-500 to-blue-500 p-3 rounded-xl">
                    <FileText className="text-white" size={24} />
                </div>
                <div className="ml-4">
                    <h3 className="text-xl font-bold text-gray-800">Generate Report</h3>
                    <p className="text-sm text-gray-600">Download your complete diagnostic report</p>
                </div>
            </div>
            
            <div className="bg-white rounded-xl p-4 mb-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <span className="text-gray-500 font-medium">Top Diagnosis:</span>
                        <p className="text-gray-800 font-semibold">{formatDiseaseName(topDiseases[0][0])}</p>
                    </div>
                    <div>
                        <span className="text-gray-500 font-medium">Confidence:</span>
                        <p className="text-gray-800 font-semibold">{(topDiseases[0][1] * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                        <span className="text-gray-500 font-medium">Symptoms Analyzed:</span>
                        <p className="text-gray-800 font-semibold">{Object.keys(parsedSymptoms).length}</p>
                    </div>
                    <div>
                        <span className="text-gray-500 font-medium">Questions Asked:</span>
                        <p className="text-gray-800 font-semibold">{history.length}</p>
                    </div>
                </div>
            </div>
            
            <div className="flex space-x-3">
                <button
                    onClick={generatePDF}
                    className="flex-1 bg-gradient-to-r from-teal-600 to-blue-600 text-white py-3 rounded-lg font-semibold hover:from-teal-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl flex items-center justify-center space-x-2"
                >
                    <Download size={20} />
                    <span>Download PDF</span>
                </button>
                <button
                    onClick={printReport}
                    className="px-6 py-3 border-2 border-teal-600 text-teal-600 rounded-lg font-semibold hover:bg-teal-50 transition-all flex items-center justify-center space-x-2"
                >
                    <Printer size={20} />
                    <span>Print</span>
                </button>
            </div>
            
            <p className="text-xs text-gray-500 mt-4 text-center italic">
                Report includes diagnosis results, symptoms, and session summary with official stamp
            </p>
        </div>
    );
};

export default ReportGenerator;
