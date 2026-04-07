import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 600000, // 10 min – long evaluations
})

export const getHealth   = ()       => api.get('/health').then(r => r.data)
export const getModels   = ()       => api.get('/models').then(r => r.data.models)
export const getTasks    = ()       => api.get('/tasks').then(r => r.data.tasks)
export const runEpisode  = (body)   => api.post('/run-episode', body).then(r => r.data)
export const runEvaluate = (body)   => api.post('/evaluate', body).then(r => r.data)
export const runLLM      = (evalResult) =>
  api.post('/llm-analyze', { evaluation: evalResult }).then(r => r.data)

export default api
