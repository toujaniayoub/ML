<?php
session_start();
if (!isset($_SESSION['user_id'])) {
    header("Location: auth.php");
    exit();
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Starter Page</title>
</head>
<body>
    <h1>Welcome to SmartPredict</h1>
    <p>Here you can predict smartphone prices and features.</p>
    <!-- Add your prediction features here -->
</body>
</html>
