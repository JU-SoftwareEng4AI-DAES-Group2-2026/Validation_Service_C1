terraform {
  #############################################################
  ## AFTER RUNNING TERRAFORM APPLY (WITH LOCAL BACKEND)
  ## YOU WILL UNCOMMENT THIS CODE THEN RERUN TERRAFORM INIT
  ## TO SWITCH FROM LOCAL BACKEND TO REMOTE AWS BACKEND
  #############################################################
  #backend "s3" {
  #  bucket         = "devops-fh-tf-state" # REPLACE WITH YOUR BUCKET NAME
  #  key            = "devops-fh/terraform.tfstate"
  #  region         = "us-east-1"
  #  dynamodb_table = "terraform-state-locking"
  #  encrypt        = true
  #}

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=4.46.0"
    }
    azapi = {
      source  = "Azure/azapi"
      version = "~> 1.13"   # or latest
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.37.1"
    }
  }
}

#provider "kubernetes" {
#  host                   = azurerm_kubernetes_cluster.aks.kube_config[0].host
#  client_certificate     = base64decode(azurerm_kubernetes_cluster.aks.kube_config[0].client_certificate)
#  client_key             = base64decode(azurerm_kubernetes_cluster.aks.kube_config[0].client_key)
#  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.aks.kube_config[0].cluster_ca_certificate)
#}


provider "azapi" {
}

provider "azurerm" {
  #resource_provider_registrations = "none" # This is only required when the User, Service Principal, or Identity running Terraform lacks the permissions to register Azure Resource Providers.
  features {}
  resource_provider_registrations = "all"
}

resource "azurerm_resource_group" "rg-project" {
  name     = "daes-group2-2026-c1"
  location = "denmarkeast"
}

data "azurerm_client_config" "current" {}

## Register Microsoft.App
#resource "azapi_resource_action" "register_microsoft_app" {
#  type        = "Microsoft.Resources/providers@2021-04-01"
#  resource_id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/providers/Microsoft.App"
#  action      = "register"
#  method      = "POST"
#}

## Often needed alongside Container Apps
#resource "azapi_resource_action" "register_operational_insights" {
#  type        = "Microsoft.Resources/providers@2021-04-01"
#  resource_id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/providers/Microsoft.OperationalInsights"
#  action      = "register"
#  method      = "POST"
#}

## Safe to register too
#resource "azapi_resource_action" "register_container_registry" {
#  type        = "Microsoft.Resources/providers@2021-04-01"
#  resource_id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/providers/Microsoft.ContainerRegistry"
#  action      = "register"
#  method      = "POST"
#}

#resource "azurerm_role_assignment" "acr_pull_ca" {
#  scope                = azurerm_container_registry.acr.id
#  role_definition_name = "AcrPull"
#  principal_id         = azurerm_container_app.api.identity[0].principal_id
#}



resource "random_integer" "ri" {
  min = 10000
  max = 99999
}


resource "azurerm_service_plan" "app_service_plan" {
  name                = "daes-c1-app-plan"
  resource_group_name = azurerm_resource_group.rg-project.name
  location            = azurerm_resource_group.rg-project.location

  os_type  = "Linux"
  sku_name = "B1"
}

resource "azurerm_linux_web_app" "c1_api" {
  name                = "daes-c1-api"
  resource_group_name = azurerm_resource_group.rg-project.name
  location            = azurerm_resource_group.rg-project.location
  service_plan_id     = azurerm_service_plan.app_service_plan.id

  https_only = true

  identity {
    type = "SystemAssigned"
  }

  site_config {
    always_on = true

    application_stack {
      docker_image_name = "ju-softwareeng4ai-daes-group2-2026/validation-service:latest"
      docker_registry_url = "https://ghcr.io"
    }

    health_check_path = "/health"
    health_check_eviction_time_in_min = 5
  }

  app_settings = {
    WEBSITES_PORT = "8000"
    WEBSITES_ENABLE_APP_SERVICE_STORAGE = "false"
  }
}

output "c1_webapp_url" {
  value = "https://${azurerm_linux_web_app.c1_api.default_hostname}"
}


#resource "azurerm_application_insights" "validation_service_insights" {
#  name                = "validation-service-ai"
#  location            = azurerm_resource_group.rg-project.location
#  resource_group_name = azurerm_resource_group.rg-project.name
#  application_type    = "web"
#}

#resource "azurerm_service_plan" "project-func-plan" {
#  location            = azurerm_resource_group.rg-project.location
#  name                = "validation-plan"
#  os_type             = "Linux"
#  resource_group_name = azurerm_resource_group.rg-project.name
#  sku_name            = "Y1"
#}


#resource "azurerm_container_app_environment" "container-env" {
# name                = "aca-env"
# resource_group_name = azurerm_resource_group.rg-project.name
# location            = azurerm_resource_group.rg-project.location
#
# depends_on = [
#   azapi_resource_action.register_microsoft_app,
#   azapi_resource_action.register_container_registry,
#   azapi_resource_action.register_operational_insights
# ]
#}
#
#resource "azurerm_container_registry" "acr" {
# location            = azurerm_resource_group.rg-project.location
# name                = "daesgroup2c12026"
# resource_group_name = azurerm_resource_group.rg-project.name
# sku                 = "Standard"
# admin_enabled       = true
# anonymous_pull_enabled = true
#}


