const cameraForm =
    document.getElementById(
        "ip-camera-form"
    );


const cameraUrlInput =
    document.getElementById(
        "camera-url"
    );


const startButton =
    document.getElementById(
        "start-ip-camera"
    );


const stopButton =
    document.getElementById(
        "stop-ip-camera"
    );


const clearButton =
    document.getElementById(
        "clear-ip-results"
    );


const cameraFrame =
    document.getElementById(
        "ip-camera-frame"
    );


const processedFrame =
    document.getElementById(
        "ip-processed-frame"
    );


const statusText =
    document.getElementById(
        "ip-camera-status"
    );


const resultTable =
    document.getElementById(
        "ip-result-table"
    );


let cameraUrl = "";

let isRunning = false;

let isProcessing = false;

let processingTimer = null;

let liveFeedTimer = null;

let isLiveFeedLoading = false;

let savedDetections = {};



cameraForm.addEventListener(
    "submit",
    async function (
        event
    ) {

        event.preventDefault();


        cameraUrl =
            cameraUrlInput.value.trim();


        if (
            !cameraUrl
        ) {

            statusText.textContent =
                "Enter a camera URL.";

            return;

        }


        statusText.textContent =
            "Connecting to camera...";


        const response =
            await fetch(
                "/connect-ip-camera",
                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            {
                                camera_url:
                                    cameraUrl
                            }
                        )
                }
            );


        const data =
            await response.json();


        if (
            !response.ok
        ) {

            statusText.textContent =
                data.detail
                || "Could not connect.";

            return;

        }


        statusText.textContent =
            "Camera connected.";


        startButton.disabled =
            false;

    }
);



startButton.addEventListener(
    "click",
    function () {

        if (
            !cameraUrl
        ) {

            return;

        }


        isRunning =
            true;


        startButton.disabled =
            true;


        stopButton.disabled =
            false;


        statusText.textContent =
            "ANPR is running...";


        liveFeedTimer =
            setInterval(
                updateLiveFeed,
                200
            );


        processingTimer =
            setInterval(
                processCameraFrame,
                1500
            );


        updateLiveFeed();

        processCameraFrame();

    }
);



stopButton.addEventListener(
    "click",
    function () {

        stopCamera();

    }
);

async function updateLiveFeed() {

    if (
        !isRunning
        || isLiveFeedLoading
    ) {

        return;

    }


    isLiveFeedLoading =
        true;


    try {

        const response =
            await fetch(
                "/ip-camera-live-frame"
            );


        const data =
            await response.json();


        if (
            !response.ok
        ) {

            throw new Error(
                data.detail
                || "Could not load live frame."
            );

        }


        cameraFrame.src =
            "data:image/jpeg;base64,"
            + data.image;

    }

    catch (
        error
    ) {

        console.error(
            "Live feed error:",
            error
        );

    }

    finally {

        isLiveFeedLoading =
            false;

    }

}

async function processCameraFrame() {

    if (
        !isRunning
        || isProcessing
    ) {

        return;

    }


    isProcessing =
        true;


    try {

        const response =
            await fetch(
                "/process-ip-camera-frame",
                {
                    method:
                        "POST"
                }
            );


        const data =
            await response.json();


        if (
            !response.ok
        ) {

            throw new Error(
                data.detail
                || "Camera processing failed."
            );

        }



        processedFrame.src =
            "data:image/jpeg;base64,"
            + data.processed_image;


        updateResults(
            data.results
        );


        statusText.textContent =
            data.status;

    }

    catch (
        error
    ) {

        console.error(
            error
        );


        statusText.textContent =
            "Camera processing error: "
            + error.message;

    }

    finally {

        isProcessing =
            false;

    }

}



function updateResults(
    results
) {

    if (
        results
        && results.length > 0
    ) {

        results.forEach(
            result => {

                const plate =
                    result.license_plate;


                if (
                    plate
                    && plate !== "Reading..."
                    && plate !== "Unknown"
                    && plate !== "Not detected"
                ) {

                    savedDetections[
                        plate
                    ] = result;

                }

            }
        );

    }


    resultTable.innerHTML =
        "";


    const savedResults =
        Object.values(
            savedDetections
        );


    if (
        savedResults.length === 0
    ) {

        resultTable.innerHTML =
            `
            <tr>

                <td colspan="3">

                    Waiting for a valid
                    licence plate...

                </td>

            </tr>
            `;

        return;

    }


    savedResults.forEach(
        result => {

            const row =
                document.createElement(
                    "tr"
                );


            row.innerHTML =
                `
                <td>

                    ${result.vehicle_id}

                </td>

                <td>

                    ${result.license_plate}

                </td>

                <td>

                    ${result.number_of_readings}

                </td>
                `;


            resultTable.appendChild(
                row
            );

        }
    );

}



function stopCamera() {

    isRunning =
        false;
    fetch(
        "/disconnect-ip-camera",
        {
            method:
                "POST"
        }
    )
    .catch(
        error => {

            console.error(
                error
            );

        }
    );

    if (
        processingTimer
    ) {

        clearInterval(
            processingTimer
        );


        processingTimer =
            null;

    }

    if (
        liveFeedTimer
    ) {

        clearInterval(
            liveFeedTimer
        );

        liveFeedTimer =
            null;

    }

    startButton.disabled =
        false;


    stopButton.disabled =
        true;


    statusText.textContent =
        "Camera stopped.";

}



clearButton.addEventListener(
    "click",
    function () {

        savedDetections =
            {};


        resultTable.innerHTML =
            `
            <tr>

                <td colspan="3">

                    Results cleared.

                </td>

            </tr>
            `;

    }
);



window.addEventListener(
    "beforeunload",
    stopCamera
);