// Chart initialization and management

function initializeCharts() {
    initializeWeatherChart();
    initializeSoilHealthChart();
    initializePestRiskChart();
    initializeYieldPredictionChart();
}

function initializeWeatherChart() {
    const ctx = document.getElementById('weatherChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Temperature (°C)',
                data: [28, 30, 32, 31, 29, 27, 26],
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: 'Rainfall (mm)',
                data: [5, 8, 12, 3, 0, 0, 2],
                borderColor: '#4ecdc4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function initializeSoilHealthChart() {
    const ctx = document.getElementById('soilHealthChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Nitrogen', 'Phosphorus', 'Potassium', 'pH Level', 'Organic Matter'],
            datasets: [{
                label: 'Current Levels',
                data: [75, 60, 85, 70, 65],
                backgroundColor: 'rgba(46, 125, 50, 0.2)',
                borderColor: 'rgba(46, 125, 50, 1)',
                pointBackgroundColor: 'rgba(46, 125, 50, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(46, 125, 50, 1)'
            }, {
                label: 'Optimal Levels',
                data: [80, 80, 80, 75, 70],
                backgroundColor: 'rgba(104, 159, 56, 0.2)',
                borderColor: 'rgba(104, 159, 56, 1)',
                pointBackgroundColor: 'rgba(104, 159, 56, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(104, 159, 56, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            }
        }
    });
}

function initializePestRiskChart() {
    const ctx = document.getElementById('pestRiskChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [60, 25, 15],
                backgroundColor: [
                    'rgba(76, 175, 80, 0.8)',
                    'rgba(255, 152, 0, 0.8)',
                    'rgba(244, 67, 54, 0.8)'
                ],
                borderColor: [
                    'rgba(76, 175, 80, 1)',
                    'rgba(255, 152, 0, 1)',
                    'rgba(244, 67, 54, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            },
            cutout: '70%'
        }
    });
}

function initializeYieldPredictionChart() {
    const ctx = document.getElementById('yieldPredictionChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Current', 'With Recommendations', 'Optimal'],
            datasets: [{
                label: 'Yield Prediction (kg/hectare)',
                data: [2500, 3200, 4000],
                backgroundColor: [
                    'rgba(104, 159, 56, 0.6)',
                    'rgba(76, 175, 80, 0.6)',
                    'rgba(46, 125, 50, 0.6)'
                ],
                borderColor: [
                    'rgba(104, 159, 56, 1)',
                    'rgba(76, 175, 80, 1)',
                    'rgba(46, 125, 50, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Yield (kg/hectare)'
                    }
                }
            }
        }
    });
}

// Export functions for global access
window.initializeCharts = initializeCharts;
window.initializeWeatherChart = initializeWeatherChart;
window.initializeSoilHealthChart = initializeSoilHealthChart;
window.initializePestRiskChart = initializePestRiskChart;
window.initializeYieldPredictionChart = initializeYieldPredictionChart;