# Clear workspace
rm(list = ls())

# 1. LOAD DATASET
# Load built-in dataset
data(airquality)

# Display structure and summary
str(airquality)
summary(airquality)

# Display first rows
head(airquality)

# Handle missing values
airquality_clean <- na.omit(airquality)

# 2. BAR PLOT
# Calculate average Ozone by Month
avg_ozone <- tapply(airquality$Ozone, airquality$Month, mean, na.rm = TRUE)

# Set plotting area (2 plots side by side)
par(mfrow = c(1, 2))

# Vertical Bar Plot
barplot(avg_ozone,
        main = "Average Ozone per Month",
        xlab = "Month",
        ylab = "Ozone Level",
        col = rainbow(length(avg_ozone)),
        border = "black")

# Add values on bars
text(x = seq_along(avg_ozone),
     y = avg_ozone,
     label = round(avg_ozone, 1),
     pos = 3)

# Horizontal Bar Plot
barplot(avg_ozone,
        main = "Horizontal Bar Plot of Ozone",
        xlab = "Ozone Level",
        col = heat.colors(length(avg_ozone)),
        horiz = TRUE)

# 3. HISTOGRAM
# Reset plotting area
par(mfrow = c(1, 1))

# Histogram with density curve
hist(airquality_clean$Temp,
     main = "Histogram of Temperature",
     xlab = "Temperature",
     col = "orange",
     border = "black",
     xlim = c(50, 100),
     freq = FALSE)

# Add density line
lines(density(airquality_clean$Temp), col = "blue", lwd = 2)

# 4. BOX PLOT
# Multiple boxplots
boxplot(airquality_clean,
        main = "Boxplot of Airquality Dataset",
        col = c("red", "green", "blue", "yellow", "purple", "cyan"),
        las = 2)

# Boxplot grouped by Month
boxplot(Ozone ~ Month,
        data = airquality_clean,
        main = "Ozone Levels by Month",
        xlab = "Month",
        ylab = "Ozone",
        col = rainbow(5))

# 5. SCATTER PLOT
# Scatter plot with regression line
plot(airquality_clean$Wind, airquality_clean$Ozone,
     main = "Scatter Plot: Wind vs Ozone",
     xlab = "Wind",
     ylab = "Ozone",
     col = "blue",
     pch = 19)

# Add regression line
model <- lm(Ozone ~ Wind, data = airquality_clean)
abline(model, col = "red", lwd = 2)

# Add grid
grid()

# 6. HEAT MAP
# Convert data to matrix
aq_matrix <- as.matrix(airquality_clean)

# Generate Heatmap
heatmap(aq_matrix,
        main = "Heatmap of Airquality Dataset",
        col = colorRampPalette(c("blue", "yellow", "red"))(100),
        scale = "column")

# 7. PAIR PLOT (EXTRA VISUALIZATION)
# Scatter plot matrix
pairs(airquality_clean,
      main = "Pairwise Scatter Plot Matrix",
      col = "darkgreen",
      pch = 19)
