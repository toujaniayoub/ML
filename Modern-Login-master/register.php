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
    $name = trim($_POST['name']);
    $email = trim($_POST['email']);
    $password = trim($_POST['password']);

    // Validate inputs
    if (!empty($name) && !empty($email) && !empty($password)) {
        // Hash the password
        $hashed_password = password_hash($password, PASSWORD_DEFAULT);

        // Insert into the database
        try {
            $sql = "INSERT INTO users (name, email, password) VALUES (:name, :email, :password)";
            $stmt = $pdo->prepare($sql);
            $stmt->execute([
                ':name' => $name,
                ':email' => $email,
                ':password' => $hashed_password
            ]);

            // Redirect to success page
            header("Location: sucess_register.html");
            exit;
        } catch (PDOException $e) {
            echo "Error: " . $e->getMessage();
        }
    } else {
        echo "All fields are required!";
    }
} else {
    echo "Invalid request method.";
}
?>
