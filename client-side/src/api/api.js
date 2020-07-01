/**
 * This scripts provides some basic functions for making GET and POST requests to the server-side and simplifies the response 
 */

 /**
  * Base URL for the server-side, e.g. https://localhost:5000
  * @param {string} baseUrl 
  */
let requestWrapper = (baseUrl) => {
    let request = {}; // Base request object for GET and POST requests
    let httpRequest = async (url, method, body) => {
        let fullUrl = '/api' + url;
        let httpParams = {
            method: method,
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            }
        };
    
        if (method && method.toUpperCase() === 'POST') {
            httpParams.body = JSON.stringify(body); // necessary otherwise Flask refuses the request
        }
    
        let response = {};
        try {
            response = await fetch(fullUrl, httpParams);
            return response.json();
        } catch (e) {
            return e;
        }
    };

    request.get = async (url) => {
        const method = 'GET';
        return await httpRequest(url, method, null);
    };

    request.post = async (url, body) => {
        const method = 'POST';
        return await httpRequest(url, method, body);
    };

    return request;
}

export default requestWrapper;