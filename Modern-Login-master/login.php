<?php
// Start session
session_start();

// Database connection (replace with your credentials)
$host = 'localhost';
$dbname = 'user_auth';
$user = 'root';
$password = '';

try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbname", $user, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Database connection failed: " . $e->getMessage());
}

// Check if the form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = trim($_POST['email']);
    $password = trim($_POST['password']);

    // Validate inputs
    if (!empty($email) && !empty($password)) {
        // Check if the user exists
        $sql = "SELECT * FROM users WHERE email = :email";
        $stmt = $pdo->prepare($sql);
        $stmt->execute([':email' => $email]);
        $user = $stmt->fetch(PDO::FETCH_ASSOC);

        if ($user && password_verify($password, $user['password'])) {
            // Redirect to login success page
            header("Location: sucess.html");
            exit;
        } else {
            echo "Invalid email or password.";
        }
    } else {
        echo "All fields are required!";
    }
} else {
    echo "Invalid request method.";
}
?>
