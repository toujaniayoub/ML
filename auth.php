<?php
session_start();
include 'db_connection.php';

if (isset($_POST['login'])) {
    $email = $_POST['email'];
    $password = $_POST['password'];
    
    $query = "SELECT * FROM users WHERE email = '$email'";
    $result = mysqli_query($conn, $query);
    
    if (mysqli_num_rows($result) > 0) {
        $user = mysqli_fetch_assoc($result);
        if (password_verify($password, $user['password'])) {
            $_SESSION['user_id'] = $user['id'];
            header("Location: starter-page.php");
            exit();
        } else {
            $error = "Invalid credentials.";
        }
    } else {
        $error = "No account found with this email.";
    }
}

if (isset($_POST['signup'])) {
    $name = $_POST['name'];
    $email = $_POST['email'];
    $password = password_hash($_POST['password'], PASSWORD_DEFAULT);

    $check_email = "SELECT * FROM users WHERE email = '$email'";
    $check_result = mysqli_query($conn, $check_email);

    if (mysqli_num_rows($check_result) > 0) {
        $error = "Email is already taken.";
    } else {
        $signup_query = "INSERT INTO users (name, email, password) VALUES ('$name', '$email', '$password')";
        if (mysqli_query($conn, $signup_query)) {
            $_SESSION['user_id'] = mysqli_insert_id($conn);
            header("Location: starter-page.php");
            exit();
        } else {
            $error = "Error signing up.";
        }
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login / Signup</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="auth-container">
        <h2>Login or Signup</h2>
        
        <!-- Login Form -->
        <form action="auth.php" method="POST" id="login-form">
            <input type="email" name="email" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit" name="login">Login</button>
        </form>
        
        <!-- Signup Form -->
        <form action="auth.php" method="POST" id="signup-form">
            <input type="text" name="name" placeholder="Name" required>
            <input type="email" name="email" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit" name="signup">Sign Up</button>
        </form>

        <?php if (isset($error)) { echo "<p>$error</p>"; } ?>

        <button onclick="toggleForm()">Switch to <?php echo isset($_POST['login']) ? 'Signup' : 'Login'; ?></button>
    </div>

    <script>
        function toggleForm() {
            const loginForm = document.getElementById('login-form');
            const signupForm = document.getElementById('signup-form');
            if (loginForm.style.display === 'none') {
                loginForm.style.display = 'block';
                signupForm.style.display = 'none';
            } else {
                loginForm.style.display = 'none';
                signupForm.style.display = 'block';
            }
        }

        // Initially show login form
        document.getElementById('signup-form').style.display = 'none';
    </script>
</body>
</html>
