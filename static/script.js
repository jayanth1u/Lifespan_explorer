// Global variables
const countrySelect = document.getElementById('country-select');
const ctx = document.getElementById('life-chart').getContext('2d');
const loading = document.getElementById('loading');
const chartSection = document.getElementById('chart-section');
const statsSection = document.getElementById('stats-section');
const chartTitle = document.getElementById('chart-title');
const chartInfo = document.getElementById('chart-info');

let chart;
let countriesData = [];

// Chart configuration
const chartConfig = {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Life Expectancy (years)',
            data: [],
            borderColor: '#3498db',
            backgroundColor: 'rgba(52, 152, 219, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#3498db',
            pointBorderColor: '#2980b9',
            pointBorderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8,
            pointHoverBackgroundColor: '#e74c3c',
            pointHoverBorderColor: '#c0392b'
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    color: '#2c3e50'
                }
            },
            tooltip: {
                backgroundColor: 'rgba(44, 62, 80, 0.9)',
                titleColor: '#ecf0f1',
                bodyColor: '#ecf0f1',
                borderColor: '#3498db',
                borderWidth: 1,
                cornerRadius: 8,
                displayColors: false,
                callbacks: {
                    title: function(context) {
                        return `Year: ${context[0].label}`;
                    },
                    label: function(context) {
                        return `Life Expectancy: ${context.parsed.y.toFixed(1)} years`;
                    }
                }
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Year',
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    color: '#2c3e50'
                },
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                },
                ticks: {
                    color: '#7f8c8d',
                    font: {
                        size: 12
                    }
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Life Expectancy (years)',
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    color: '#2c3e50'
                },
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                },
                ticks: {
                    color: '#7f8c8d',
                    font: {
                        size: 12
                    }
                }
            }
        },
        interaction: {
            intersect: false,
            mode: 'index'
        },
        animation: {
            duration: 1000,
            easing: 'easeInOutQuart'
        }
    }
};

// Utility functions
function showLoading() {
    loading.style.display = 'block';
    chartSection.style.display = 'none';
    statsSection.style.display = 'none';
}

function hideLoading() {
    loading.style.display = 'none';
}

function showError(message) {
    console.error('Error:', message);
    hideLoading();
    // You could add a toast notification here
}

function formatNumber(num, decimals = 1) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return Number(num).toFixed(decimals);
}

function formatLargeNumber(num) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toFixed(0);
}

function calculateStats(data) {
    if (!data || data.length === 0) return null;

    const lifeExpectancies = data.map(d => d['Life expectancy ']).filter(val => val != null && !isNaN(val));
    const gdpValues = data.map(d => d.GDP).filter(val => val != null && !isNaN(val));
    const schoolingValues = data.map(d => d.Schooling).filter(val => val != null && !isNaN(val));

    const avgLifeExpectancy = lifeExpectancies.length > 0 
        ? lifeExpectancies.reduce((a, b) => a + b, 0) / lifeExpectancies.length 
        : 0;

    const avgGDP = gdpValues.length > 0 
        ? gdpValues.reduce((a, b) => a + b, 0) / gdpValues.length 
        : 0;

    const avgSchooling = schoolingValues.length > 0 
        ? schoolingValues.reduce((a, b) => a + b, 0) / schoolingValues.length 
        : 0;

    // Calculate trend (change from first to last available data point)
    let trendChange = 0;
    if (lifeExpectancies.length >= 2) {
        const firstValue = lifeExpectancies[0];
        const lastValue = lifeExpectancies[lifeExpectancies.length - 1];
        trendChange = lastValue - firstValue;
    }

    return {
        avgLifeExpectancy,
        avgGDP,
        avgSchooling,
        trendChange,
        dataPoints: lifeExpectancies.length
    };
}

function updateStats(data) {
    const stats = calculateStats(data);
    
    if (!stats) {
        document.getElementById('avg-life-expectancy').textContent = '--';
        document.getElementById('trend-change').textContent = '--';
        document.getElementById('avg-gdp').textContent = '--';
        document.getElementById('avg-schooling').textContent = '--';
        return;
    }

    document.getElementById('avg-life-expectancy').textContent = formatNumber(stats.avgLifeExpectancy) + ' years';
    
    const trendElement = document.getElementById('trend-change');
    const trendValue = stats.trendChange;
    if (trendValue > 0) {
        trendElement.textContent = '+' + formatNumber(trendValue) + ' years';
        trendElement.style.color = '#27ae60';
    } else if (trendValue < 0) {
        trendElement.textContent = formatNumber(trendValue) + ' years';
        trendElement.style.color = '#e74c3c';
    } else {
        trendElement.textContent = formatNumber(trendValue) + ' years';
        trendElement.style.color = '#f39c12';
    }
    
    document.getElementById('avg-gdp').textContent = '$' + formatLargeNumber(stats.avgGDP);
    document.getElementById('avg-schooling').textContent = formatNumber(stats.avgSchooling) + ' years';
}

// Load countries on page load
async function loadCountries() {
    try {
        console.log('Loading countries...');
        const response = await fetch('/countries');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const countries = await response.json();
        console.log('Countries loaded:', countries.length);
        
        if (!countries || countries.length === 0) {
            throw new Error('No countries data received');
        }
        
        countriesData = countries;
        
        // Clear existing options except the first one
        countrySelect.innerHTML = '<option value="">--Choose a country--</option>';
        
        // Add countries to select
        countries.forEach(country => {
            const option = document.createElement('option');
            option.value = country;
            option.textContent = country;
            countrySelect.appendChild(option);
        });
        
        console.log('Countries populated in select');
        
    } catch (error) {
        console.error('Error loading countries:', error);
        showError('Failed to load countries. Please check your data file and server.');
        countrySelect.innerHTML = '<option value="">Error loading countries</option>';
    }
}

// Load data for selected country
async function loadCountryData(country) {
    if (!country) {
        chartSection.style.display = 'none';
        statsSection.style.display = 'none';
        return;
    }

    showLoading();
    
    try {
        console.log('Loading data for:', country);
        const response = await fetch(`/data?country=${encodeURIComponent(country)}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Data loaded for', country, ':', data.length, 'records');
        
        if (!data || data.length === 0) {
            throw new Error('No data available for this country');
        }

        // Prepare chart data
        const years = data.map(d => d.Year);
        const lifeExpectancies = data.map(d => d['Life expectancy ']);
        
        // Destroy existing chart
        if (chart) {
            chart.destroy();
        }
        
        // Create new chart
        const newConfig = { ...chartConfig };
        newConfig.data.labels = years;
        newConfig.data.datasets[0].data = lifeExpectancies;
        
        chart = new Chart(ctx, newConfig);
        
        // Update titles and info
        chartTitle.textContent = `Life Expectancy Trends - ${country}`;
        chartInfo.textContent = `Data from ${Math.min(...years)} to ${Math.max(...years)} (${data.length} data points)`;
        
        // Update statistics
        updateStats(data);
        
        // Show sections
        hideLoading();
        chartSection.style.display = 'block';
        statsSection.style.display = 'block';
        
        // Smooth scroll to chart
        chartSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        console.error('Error loading country data:', error);
        showError(`Failed to load data for ${country}. ${error.message}`);
    }
}

// Event listeners
countrySelect.addEventListener('change', (event) => {
    const selectedCountry = event.target.value;
    console.log('Country selected:', selectedCountry);
    loadCountryData(selectedCountry);
});

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing app...');
    loadCountries();
});

// Add some debugging
window.addEventListener('load', () => {
    console.log('Window loaded');
});

// Error handling for unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});