const helpers = {};

// Helper function for building date and time histograms with current data retrieved
let dateHistogram = (instances) => {
    return instances.reduce((prev, item) => {
        if (prev[item.DateTime] === undefined) {
            prev[item.DateTime] = [];
        }

        prev[item.DateTime].push({
            longitude: item.Longitude,
            latitude: item.Latitude
        });

        return prev;

    }, {});
}