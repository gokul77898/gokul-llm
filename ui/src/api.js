import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
})

export const queryAPI = async ({ query, model, top_k }) => {
  try {
    const response = await api.post('/query', {
      query,
      model,
      top_k,
    })
    return response.data
  } catch (error) {
    if (error.response) {
      // Server responded with error
      throw new Error(error.response.data.detail || error.response.data.message || 'Query failed')
    } else if (error.request) {
      // Request made but no response
      throw new Error('Backend not responding. Please ensure the API is running at ' + API_BASE_URL)
    } else {
      // Something else happened
      throw new Error(error.message || 'Query failed')
    }
  }
}

export const getModels = async () => {
  try {
    const response = await api.get('/models')
    return response.data
  } catch (error) {
    console.error('Failed to fetch models:', error)
    // Return default models if API fails
    return {
      models: ['inlegalbert', 'incaselawbert']
    }
  }
}

export const runMoE = async (text, task) => {
  try {
    const payload = { text, task, max_tokens: 256 }
    const response = await api.post('/moe-generate', payload)
    return response.data
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || error.response.data.message || 'MoE generation failed')
    } else if (error.request) {
      throw new Error('Backend not responding. Please ensure the MoE API is running at ' + API_BASE_URL)
    } else {
      throw new Error(error.message || 'MoE generation failed')
    }
  }
}

export const healthCheck = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    return { status: 'unavailable', error: error.message }
  }
}

export default api
