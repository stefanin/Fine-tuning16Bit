<?php
// Imposta l'header per rispondere in formato JSON, cruciale per la comunicazione
header('Content-Type: application/json');

// --- 1. Includi le credenziali del database ---
// Questo file contiene le variabili $servername, $username, $password, $dbname
// e, quando la riattiverai, la variabile $apiKey.
require_once 'credential.php'; 

// --- 2. Sistema di Autenticazione (TEMPORANEAMENTE DISABILITATO) ---
/*  // Per RIATTIVARE, cancella questa riga
$headers = getallheaders();
$authHeader = isset($headers['X-API-Key']) ? $headers['X-API-Key'] : null;

// La variabile $apiKey viene dal file credential.php
if ($authHeader !== $apiKey) {
    // Se la chiave API non corrisponde o è assente, nega l'accesso.
    http_response_code(401); // Codice HTTP per "Unauthorized"
    echo json_encode(['error' => 'Accesso negato. Chiave API non valida.']);
    exit(); // Termina lo script
}
*/  // E cancella questa riga


// --- 3. Connessione al Database ---
// Le variabili sono definite in 'credential.php'
$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    http_response_code(500); // Errore interno del server
    echo json_encode(['error' => 'Connessione al database fallita: ' . $conn->connect_error]);
    exit();
}

// --- 4. Esecuzione della Query ---
// Prende il parametro 'cod' dall'URL (es. ...?cod=BD710)
$codice_da_cercare = isset($_GET['cod']) ? $_GET['cod'] : '';
if (empty($codice_da_cercare)) {
    http_response_code(400); // Codice per una richiesta malformata
    echo json_encode(['error' => 'Nessun codice componente fornito.']);
    exit();
}

// Usa un prepared statement per prevenire SQL Injection
$stmt = $conn->prepare("SELECT cod, des, qta, sc FROM mag WHERE cod LIKE ?");
$param = "%" . $codice_da_cercare . "%"; // Aggiunge i wildcard per una ricerca parziale
$stmt->bind_param("s", $param);

$stmt->execute();
$result = $stmt->get_result();

$risultati = [];
if ($result->num_rows > 0) {
    // Itera su ogni riga trovata e aggiungila all'array dei risultati
    while($row = $result->fetch_assoc()) {
        $risultati[] = $row;
    }
}

// Chiudi le connessioni per liberare risorse
$stmt->close();
$conn->close();

// Stampa l'array dei risultati in formato JSON
echo json_encode($risultati);
?>