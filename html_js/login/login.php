<?php
include("connection.php");
echo("hiiii");
if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['submit'])) 
{
    $username = $_POST['username'];
    $password = $_POST['password'];

    echo("redirect2");
    $sql = "SELECT * FROM users WHERE username = '$username' AND password = '$password'";
    echo("redirect1");
    $result = mysqli_query($conn, $sql);
    $row = mysqli_fetch_array($result, MYSQLI_ASSOC);
    $count = mysqli_num_rows($result);
    echo("redirect");
    if ($count == 1) {
        header("Location:welcome.php");
        exit(); // Stop further execution
    } else {
        echo '<script>
            window.location.href = "index.php";
            alert("Login failed. Invalid username or password!!");
        </script>';
    }
echo("jharna");
}
?>
