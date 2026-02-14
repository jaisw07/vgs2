import React, { useState } from 'react';
import { Network, Activity, Grid3X3, Database, X } from 'lucide-react';

export default function ResultsPage() {
  const [expandedCard, setExpandedCard] = useState<string | null>(null);

  const visualizations = [
    {
      id: 'dataset-overview',
      icon: Database,
      color: 'from-purple-500 to-purple-600',
      title: 'Dataset Overview',
      shortDesc: 'Comprehensive view of the entire medical knowledge base structure',
      fullDesc: [
        'A comprehensive view of the entire medical dataset, showcasing the structure and relationships between diseases and symptoms. This visualization provides insight into the breadth and depth of our knowledge base, illustrating the complexity and interconnectedness of medical conditions and their presentations.'
      ],
      image: '/dataset.png'
    },
    {
      id: 'disease-similarity',
      icon: Network,
      color: 'from-teal-500 to-teal-600',
      title: 'Disease Similarity Graph',
      shortDesc: 'Visualizes relationships between diseases based on shared symptom patterns',
      fullDesc: [
        'The disease similarity graph visualizes how closely related different diseases are based on their shared symptom patterns in the dataset. Each node represents a disease, and edges represent strong similarity between two conditions—meaning they tend to occur with a similar combination of symptoms. Diseases that appear close together in the graph share more overlapping symptom profiles, while distant or unconnected diseases have very different presentations.',
        'This visualization helps users and clinicians quickly identify clusters of conditions with similar symptom patterns, supporting faster differential diagnosis and a better understanding of relationships within the medical dataset.'
      ],
      image: '/disease_similarity_graph_colour_coded.png'
    },
    {
      id: 'symptom-cooccurrence',
      icon: Activity,
      color: 'from-blue-500 to-blue-600',
      title: 'Symptom Co-occurrence Graph',
      shortDesc: 'Highlights symptoms that frequently appear together in patient data',
      fullDesc: [
        'The symptom co-occurrence graph highlights which symptoms tend to appear together in real patient data. Each node represents a symptom, and an edge between two symptoms indicates that they are frequently reported in combination. This visualization helps reveal natural symptom groupings—such as clusters typical of specific illnesses—and makes it easier to understand how symptoms relate to one another across the dataset.',
        'It provides an intuitive way to explore patterns in patient experiences and how different symptoms may be linked.'
      ],
      image: '/symptom_cooccurence_graph.png'
    },
    {
      id: 'disease-symptom-heatmap',
      icon: Grid3X3,
      color: 'from-cyan-500 to-cyan-600',
      title: 'Disease–Symptom Heatmap',
      shortDesc: 'Visual overview of symptom associations with each disease',
      fullDesc: [
        'The disease–symptom heatmap provides a visual overview of how strongly each symptom is associated with each disease in the dataset. Each row represents a disease and each column represents a symptom, with color intensity showing the likelihood of that symptom occurring for that condition. Darker or brighter cells indicate stronger associations.',
        'This heatmap makes it easy to spot key symptom patterns, compare diseases at a glance, and understand which symptoms are most informative for distinguishing between conditions. It offers an intuitive, high-level view of the underlying probability relationships used by the diagnostic system.'
      ],
      image: '/kbr_symptom_disease_heatmap.png'
    }
    
  ];

  const toggleCard = (id: string) => {
    setExpandedCard((prev) => (prev === id ? null : id));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-teal-50 to-cyan-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
            Dataset Visualizations
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Explore the relationships and patterns within our medical knowledge base
          </p>
        </div>

        {/* Grid of Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {visualizations.map((viz) => {
            const Icon = viz.icon;
            const isExpanded = expandedCard === viz.id;
            
            return (
              <div
                key={viz.id}
                className={`bg-gradient-to-br ${isExpanded ? 'from-gray-900 to-gray-800' : 'from-gray-800 to-gray-900'} rounded-2xl shadow-xl transition-all duration-300 cursor-pointer overflow-hidden ${
                  isExpanded ? 'md:col-span-2' : ''
                }`}
                onClick={() => !isExpanded && toggleCard(viz.id)}
              >
                {/* Card Header - Always Visible */}
                <div className="p-8">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-4">
                      <div className={`bg-gradient-to-br ${viz.color} w-14 h-14 rounded-xl flex items-center justify-center flex-shrink-0`}>
                        <Icon className="text-white" size={28} />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-white mb-2">
                          {viz.title}
                        </h2>
                        {!isExpanded && (
                          <p className="text-gray-300 text-sm">
                            {viz.shortDesc}
                          </p>
                        )}
                      </div>
                    </div>
                    {isExpanded && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleCard(viz.id);
                        }}
                        className="text-gray-400 hover:text-white transition-colors"
                      >
                        <X size={24} />
                      </button>
                    )}
                  </div>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div className="mt-6 space-y-6">
                      {/* Description */}
                      <div className="space-y-4">
                        {viz.fullDesc.map((paragraph, idx) => (
                          <p key={idx} className="text-gray-300 leading-relaxed">
                            {paragraph}
                          </p>
                        ))}
                      </div>

                      {/* Image */}
                      <div className="bg-white rounded-xl p-6 flex justify-center">
                        <img
                          src={viz.image}
                          alt={viz.title}
                          className="max-w-full h-auto rounded-lg shadow-lg"
                        />
                      </div>
                    </div>
                  )}

                  {/* Click Hint */}
                  {!isExpanded && (
                    <div className="mt-4 text-center">
                      <span className="text-gray-400 text-sm">Click to expand</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="text-center">
          <p className="text-sm text-gray-500">
            All visualizations are generated from the medical knowledge base and updated at the time of model making.
          </p>
        </div>
      </div>
    </div>
  );
}